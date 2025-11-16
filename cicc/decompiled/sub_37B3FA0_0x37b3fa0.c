// Function: sub_37B3FA0
// Address: 0x37b3fa0
//
__int64 __fastcall sub_37B3FA0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v3; // r12
  _QWORD *v4; // r14
  unsigned int v7; // r8d
  __int64 v8; // r13
  int v9; // eax
  __int64 v10; // r9
  __int64 v11; // r9
  __int64 i; // rbx
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64, unsigned __int16); // r10
  __int64 v17; // rax
  int v19; // [rsp+0h] [rbp-40h]
  unsigned int v20; // [rsp+4h] [rbp-3Ch]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD **)(a2 + 40);
  v4 = &v3[2 * *(unsigned int *)(a2 + 48)];
  if ( v3 == v4 )
  {
    return 0;
  }
  else
  {
    v7 = 0;
    do
    {
      if ( (*v3 & 6) == 0 )
      {
        v8 = *(_QWORD *)(*v3 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v8 )
        {
          v9 = *(_DWORD *)(v8 + 24);
          if ( v9 == 50 )
          {
LABEL_15:
            ++v7;
          }
          else if ( v9 < 0 )
          {
            v10 = *(unsigned int *)(v8 + 68);
            if ( (_DWORD)v10 )
            {
              v11 = 16 * v10;
              for ( i = 0; v11 != i; i += 16 )
              {
                v13 = *(unsigned __int16 *)(*(_QWORD *)(v8 + 48) + i);
                if ( (_WORD)v13 )
                {
                  v14 = *(_QWORD *)(a1 + 136);
                  v15 = *(_QWORD *)(v14 + 8LL * (unsigned __int16)v13 + 112);
                  if ( v15 )
                  {
                    v16 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v14 + 552LL);
                    if ( v16 == sub_2EC09E0 )
                    {
                      if ( a3 == *(unsigned __int16 *)(*(_QWORD *)v15 + 24LL) )
                        goto LABEL_15;
                    }
                    else
                    {
                      v19 = a3;
                      v20 = v7;
                      v21 = v11;
                      v17 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v16)(v14, v13, 0);
                      a3 = v19;
                      v7 = v20;
                      v11 = v21;
                      if ( v19 == *(unsigned __int16 *)(*(_QWORD *)v17 + 24LL) )
                        goto LABEL_15;
                    }
                  }
                }
              }
            }
          }
        }
      }
      v3 += 2;
    }
    while ( v4 != v3 );
  }
  return v7;
}
