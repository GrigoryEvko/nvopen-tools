// Function: sub_2041DA0
// Address: 0x2041da0
//
__int64 __fastcall sub_2041DA0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v3; // r12
  _QWORD *v4; // r14
  unsigned int v6; // r8d
  __int64 v7; // r13
  __int16 v8; // ax
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 i; // rbx
  unsigned __int8 v12; // si
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 (*v15)(void); // r9
  __int64 v16; // rax
  int v18; // [rsp+0h] [rbp-40h]
  unsigned int v19; // [rsp+4h] [rbp-3Ch]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD **)(a2 + 32);
  v4 = &v3[2 * *(unsigned int *)(a2 + 40)];
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
          if ( v8 == 47 )
          {
LABEL_15:
            ++v6;
          }
          else if ( v8 < 0 )
          {
            v9 = *(unsigned int *)(v7 + 60);
            if ( (_DWORD)v9 )
            {
              v10 = 16 * v9;
              for ( i = 0; v10 != i; i += 16 )
              {
                v12 = *(_BYTE *)(*(_QWORD *)(v7 + 40) + i);
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
                      v18 = a3;
                      v19 = v6;
                      v20 = v10;
                      v16 = v15();
                      a3 = v18;
                      v6 = v19;
                      v10 = v20;
                      if ( v18 == *(unsigned __int16 *)(*(_QWORD *)v16 + 24LL) )
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
  return v6;
}
