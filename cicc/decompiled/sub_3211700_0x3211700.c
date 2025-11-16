// Function: sub_3211700
// Address: 0x3211700
//
__int64 __fastcall sub_3211700(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // rax
  void (*v6)(void); // rax
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // r13
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned __int8 v15; // dl
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // r8d
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax

  result = a1[1];
  if ( result && *(_BYTE *)(result + 782) )
  {
    v5 = *a1;
    a1[8] = a2;
    v6 = *(void (**)(void))(v5 + 144);
    if ( v6 != nullsub_1853 )
      v6();
    result = *((unsigned int *)a1 + 114);
    v7 = a1[55];
    if ( (_DWORD)result )
    {
      v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
      {
LABEL_7:
        result = v7 + 16 * result;
        if ( v9 != (__int64 *)result && !v9[1] )
        {
          v11 = (__int64 *)sub_2E88D60(a1[8]);
          v12 = sub_B92180(*v11);
          v15 = *(_BYTE *)(v12 - 16);
          if ( (v15 & 2) != 0 )
            v16 = *(_QWORD *)(v12 - 32);
          else
            v16 = v12 - 16 - 8LL * ((v15 >> 2) & 0xF);
          result = *(_QWORD *)(v16 + 40);
          v17 = a1[4];
          if ( *(_DWORD *)(result + 32) != 3 && !v17 )
          {
            v19 = a1[2];
            v20 = *(_QWORD *)(v19 + 2480);
            v21 = v19 + 8;
            if ( !v20 )
              v20 = v21;
            v22 = sub_E6C430(v20, v10, 0, v13, v14);
            a1[4] = v22;
            result = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[1] + 224) + 208LL))(
                       *(_QWORD *)(a1[1] + 224),
                       v22,
                       0);
            v17 = a1[4];
          }
          v9[1] = v17;
        }
      }
      else
      {
        v18 = 1;
        while ( v10 != -4096 )
        {
          v8 = (result - 1) & (v18 + v8);
          v9 = (__int64 *)(v7 + 16LL * v8);
          v10 = *v9;
          if ( a2 == *v9 )
            goto LABEL_7;
          ++v18;
        }
      }
    }
  }
  return result;
}
