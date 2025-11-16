// Function: sub_25D98D0
// Address: 0x25d98d0
//
void __fastcall sub_25D98D0(__int64 *a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v4; // rdi
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  bool v10; // cc
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  char v16; // r14
  __int64 *v17; // rax
  __int64 v18; // r15
  __int64 v19; // rsi
  _QWORD *v20; // rdi
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rcx
  __int64 *v24; // rax
  __int64 v25; // [rsp-58h] [rbp-58h]
  __int64 v26; // [rsp-50h] [rbp-50h]
  __int64 v27; // [rsp-50h] [rbp-50h]
  __int64 v28; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 )
  {
    for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
    {
      while ( 1 )
      {
        v4 = *(_QWORD *)(i + 24);
        if ( *(_BYTE *)v4 == 85 )
          break;
LABEL_4:
        i = *(_QWORD *)(i + 8);
        if ( !i )
          return;
      }
      v5 = *a1;
      v6 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
      v7 = *(_QWORD *)(v4 + 32 * (1 - v6));
      v8 = *(_QWORD *)(v4 + 32 * (2 - v6));
      if ( *(_BYTE *)v7 != 17 )
      {
        v28 = *(_QWORD *)(v8 + 24);
        v13 = sub_25D8EF0(v5 + 440, &v28);
        if ( v13[15] )
        {
          v14 = v13[13];
          v15 = (__int64)(v13 + 11);
          v16 = 0;
        }
        else
        {
          v14 = *v13;
          v15 = *v13 + 16LL * *((unsigned int *)v13 + 2);
          v16 = 1;
        }
        v25 = v15;
        if ( v16 )
        {
          while ( v25 != v14 )
          {
            v17 = (__int64 *)v14;
LABEL_16:
            v18 = *a1;
            v19 = *v17;
            if ( *(_BYTE *)(*a1 + 500) )
            {
              v20 = *(_QWORD **)(v18 + 480);
              v21 = &v20[*(unsigned int *)(v18 + 492)];
              v22 = v20;
              if ( v20 != v21 )
              {
                while ( v19 != *v22 )
                {
                  if ( v21 == ++v22 )
                    goto LABEL_22;
                }
                v23 = (unsigned int)(*(_DWORD *)(v18 + 492) - 1);
                *(_DWORD *)(v18 + 492) = v23;
                *v22 = v20[v23];
                ++*(_QWORD *)(v18 + 472);
              }
            }
            else
            {
              v27 = v14;
              v24 = sub_C8CA60(v18 + 472, v19);
              v14 = v27;
              if ( v24 )
              {
                *v24 = -2;
                ++*(_DWORD *)(v18 + 496);
                ++*(_QWORD *)(v18 + 472);
              }
            }
LABEL_22:
            if ( !v16 )
            {
              v14 = sub_220EF30(v14);
              goto LABEL_14;
            }
            v14 += 16;
          }
        }
        else
        {
LABEL_14:
          if ( v25 != v14 )
          {
            v17 = (__int64 *)(v14 + 32);
            goto LABEL_16;
          }
        }
        goto LABEL_4;
      }
      v9 = *(_QWORD *)(v8 + 24);
      v10 = *(_DWORD *)(v7 + 32) <= 0x40u;
      v11 = *(_QWORD **)(v7 + 24);
      v28 = v9;
      if ( !v10 )
        v11 = (_QWORD *)*v11;
      v26 = v9;
      v12 = sub_B43CB0(v4);
      sub_25D9180(v5, v12, v26, (__int64)v11);
    }
  }
}
