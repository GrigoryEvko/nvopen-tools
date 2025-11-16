// Function: sub_26E0A90
// Address: 0x26e0a90
//
__int64 __fastcall sub_26E0A90(_QWORD *a1, _QWORD *a2, char a3)
{
  char v3; // r10
  __int64 v6; // r15
  size_t v7; // r12
  __int64 result; // rax
  __int64 v9; // rsi
  unsigned int v10; // edx
  __int64 *v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // r13
  _QWORD *v14; // r14
  __int64 i; // r12
  int v16; // ecx
  int v17; // r9d
  int *v19; // [rsp+8h] [rbp-E8h]
  size_t v20[2]; // [rsp+10h] [rbp-E0h] BYREF
  int v21[52]; // [rsp+20h] [rbp-D0h] BYREF

  v3 = a3;
  v6 = a1[3];
  v7 = a2[3];
  v19 = (int *)a2[2];
  if ( v19 )
  {
    sub_C7D030(v21);
    sub_C7D280(v21, v19, v7);
    sub_C7D290(v21, v20);
    v7 = v20[0];
    v3 = a3;
  }
  result = *(unsigned int *)(v6 + 24);
  v9 = *(_QWORD *)(v6 + 8);
  if ( (_DWORD)result )
  {
    v10 = (result - 1) & (((0xBF58476D1CE4E5B9LL * v7) >> 31) ^ (484763065 * v7));
    v11 = (__int64 *)(v9 + 24LL * v10);
    v12 = *v11;
    if ( v7 == *v11 )
    {
LABEL_5:
      result = v9 + 24 * result;
      if ( v11 != (__int64 *)result )
      {
        result = v11[2];
        if ( a2[1] == result )
        {
          v13 = a2[18];
          v14 = a2 + 16;
          if ( v14 != (_QWORD *)v13 )
          {
            do
            {
              for ( i = *(_QWORD *)(v13 + 64); v13 + 48 != i; i = sub_220EF30(i) )
                sub_26E0A90(a1, i + 48, 0);
              result = sub_220EF30(v13);
              v13 = result;
            }
            while ( v14 != (_QWORD *)result );
          }
        }
        else
        {
          if ( v3 )
            ++a1[44];
          result = a2[7];
          a1[49] += result;
        }
      }
    }
    else
    {
      v16 = 1;
      while ( v12 != -1 )
      {
        v17 = v16 + 1;
        v10 = (result - 1) & (v16 + v10);
        v11 = (__int64 *)(v9 + 24LL * v10);
        v12 = *v11;
        if ( v7 == *v11 )
          goto LABEL_5;
        v16 = v17;
      }
    }
  }
  return result;
}
