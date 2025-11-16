// Function: sub_B9AC10
// Address: 0xb9ac10
//
unsigned __int64 __fastcall sub_B9AC10(__int64 a1, unsigned int *a2, unsigned int *a3)
{
  unsigned __int64 result; // rax
  unsigned int *i; // r14
  unsigned __int64 v5; // rdx
  unsigned int *v6; // rbx
  unsigned int v7; // r15d
  unsigned int *v8; // r13
  _BOOL4 v9; // r11d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  __int64 v14; // [rsp+10h] [rbp-50h]
  _BOOL4 v15; // [rsp+1Ch] [rbp-44h]
  _QWORD *v17; // [rsp+28h] [rbp-38h]

  v17 = (_QWORD *)(a1 + 144);
  result = a1 + 16;
  if ( a2 != a3 )
  {
    for ( i = a2; a3 != i; ++i )
    {
      if ( !*(_QWORD *)(a1 + 184) )
      {
        v5 = *(unsigned int *)(a1 + 8);
        v6 = (unsigned int *)(*(_QWORD *)a1 + 4 * v5);
        if ( *(unsigned int **)a1 == v6 )
        {
          if ( v5 <= 0x1F )
          {
            v7 = *i;
LABEL_24:
            result = *(unsigned int *)(a1 + 12);
            v13 = v5 + 1;
            if ( v13 > result )
            {
              sub_C8D5F0(a1, a1 + 16, v13, 4);
              result = *(_QWORD *)a1;
              v6 = (unsigned int *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8));
            }
            *v6 = v7;
            ++*(_DWORD *)(a1 + 8);
            continue;
          }
        }
        else
        {
          v7 = *i;
          result = *(_QWORD *)a1;
          while ( *(_DWORD *)result != v7 )
          {
            result += 4LL;
            if ( v6 == (unsigned int *)result )
              goto LABEL_13;
          }
          if ( v6 != (unsigned int *)result )
            continue;
LABEL_13:
          if ( v5 <= 0x1F )
            goto LABEL_24;
          v8 = *(unsigned int **)a1;
          do
          {
            v11 = sub_B9AB10(v17, a1 + 152, v8);
            if ( v12 )
            {
              v9 = v11 || v12 == a1 + 152 || *v8 < *(_DWORD *)(v12 + 32);
              v14 = v12;
              v15 = v9;
              v10 = sub_22077B0(40);
              *(_DWORD *)(v10 + 32) = *v8;
              sub_220F040(v15, v10, v14, a1 + 152);
              ++*(_QWORD *)(a1 + 184);
            }
            ++v8;
          }
          while ( v6 != v8 );
        }
        *(_DWORD *)(a1 + 8) = 0;
      }
      result = sub_B99820((__int64)v17, i);
    }
  }
  return result;
}
