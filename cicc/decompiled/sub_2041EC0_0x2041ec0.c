// Function: sub_2041EC0
// Address: 0x2041ec0
//
__int64 __fastcall sub_2041EC0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v3; // r12
  _QWORD *v4; // r15
  unsigned int v6; // r8d
  __int64 v7; // r13
  __int16 v8; // ax
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned __int8 v12; // si
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 (*v15)(void); // r9
  __int64 v16; // rax
  __int64 v18; // [rsp+0h] [rbp-40h]
  int v19; // [rsp+8h] [rbp-38h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[2 * *(unsigned int *)(a2 + 120)];
  if ( v3 == v4 )
  {
    return 0;
  }
  else
  {
    v6 = 0;
    do
    {
      if ( (*v3 & 6) == 0 )
      {
        v7 = *(_QWORD *)(*v3 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v7 )
        {
          v8 = *(_WORD *)(v7 + 24);
          if ( v8 == 46 )
          {
LABEL_15:
            ++v6;
          }
          else if ( v8 < 0 )
          {
            v9 = *(unsigned int *)(v7 + 56);
            if ( (_DWORD)v9 )
            {
              v10 = 0;
              v11 = 40 * v9;
              do
              {
                v12 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 + *(_QWORD *)(v7 + 32)) + 40LL)
                               + 16LL * *(unsigned int *)(v10 + *(_QWORD *)(v7 + 32) + 8));
                if ( v12 )
                {
                  v13 = *(_QWORD *)(a1 + 136);
                  v14 = *(_QWORD *)(v13 + 8LL * v12 + 120);
                  if ( v14 )
                  {
                    v15 = *(__int64 (**)(void))(*(_QWORD *)v13 + 288LL);
                    if ( (char *)v15 == (char *)sub_1D45FB0 )
                    {
                      if ( a3 == *(unsigned __int16 *)(*(_QWORD *)v14 + 24LL) )
                        goto LABEL_15;
                    }
                    else
                    {
                      v19 = a3;
                      v18 = a1;
                      v20 = v6;
                      v16 = v15();
                      a3 = v19;
                      a1 = v18;
                      v6 = v20;
                      if ( v19 == *(unsigned __int16 *)(*(_QWORD *)v16 + 24LL) )
                        goto LABEL_15;
                    }
                  }
                }
                v10 += 40;
              }
              while ( v11 != v10 );
            }
          }
        }
      }
      v3 += 2;
    }
    while ( v4 != v3 );
  }
  return v6;
}
