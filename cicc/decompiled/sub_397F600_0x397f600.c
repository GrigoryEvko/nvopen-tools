// Function: sub_397F600
// Address: 0x397f600
//
__int64 __fastcall sub_397F600(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rax
  void (*v5)(void); // rax
  __int64 v6; // rcx
  int v7; // r8d
  unsigned int v8; // edx
  __int64 *v9; // r13
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rax

  result = a1[2];
  if ( *(_BYTE *)(result + 1744) )
  {
    v4 = *a1;
    a1[7] = a2;
    v5 = *(void (**)(void))(v4 + 128);
    if ( v5 != nullsub_1979 )
      v5();
    result = *((unsigned int *)a1 + 94);
    if ( (_DWORD)result )
    {
      v6 = a1[45];
      v7 = 1;
      v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
      {
LABEL_6:
        result = v6 + 16 * result;
        if ( v9 != (__int64 *)result && !v9[1] )
        {
          v11 = (__int64 *)sub_1E15F70(a1[7]);
          v12 = sub_1626D20(*v11);
          v13 = *(_DWORD *)(*(_QWORD *)(v12 + 8 * (5LL - *(unsigned int *)(v12 + 8))) + 36LL) == 3;
          result = a1[4];
          if ( !v13 && !result )
          {
            v14 = sub_38BFA60(a1[2] + 168, 1);
            a1[4] = v14;
            (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[1] + 256) + 176LL))(
              *(_QWORD *)(a1[1] + 256),
              v14,
              0);
            result = a1[4];
          }
          v9[1] = result;
        }
      }
      else
      {
        while ( v10 != -8 )
        {
          v8 = (result - 1) & (v7 + v8);
          v9 = (__int64 *)(v6 + 16LL * v8);
          v10 = *v9;
          if ( a2 == *v9 )
            goto LABEL_6;
          ++v7;
        }
      }
    }
  }
  return result;
}
