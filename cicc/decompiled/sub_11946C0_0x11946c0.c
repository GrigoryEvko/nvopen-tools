// Function: sub_11946C0
// Address: 0x11946c0
//
__int64 __fastcall sub_11946C0(__int64 a1, unsigned __int8 *a2, _BYTE *a3)
{
  unsigned int v4; // eax
  _QWORD *v5; // rcx
  unsigned int v6; // r12d
  __int64 v8; // rax
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int8 *v13; // rbx
  unsigned int v14; // eax
  __int64 v15; // r12
  int v16; // r13d
  unsigned int v17; // r15d
  _BYTE *v18; // rax
  unsigned int v19; // eax
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+8h] [rbp-38h]

  v4 = sub_BCB060(**(_QWORD **)a1);
  v21 = v4;
  if ( v4 > 0x40 )
    sub_C43690((__int64)&v20, v4, 0);
  else
    v20 = v4;
  v5 = *(_QWORD **)(a1 + 24);
  if ( *a2 == **(_DWORD **)(a1 + 8) + 29 )
  {
    v8 = *((_QWORD *)a2 - 8);
    if ( v8 )
    {
      **(_QWORD **)(a1 + 16) = v8;
      v9 = (_BYTE *)*((_QWORD *)a2 - 4);
      if ( *v9 <= 0x15u )
      {
        if ( (*v5 = v9, (v10 = *((_QWORD *)a2 + 2)) != 0) && !*(_QWORD *)(v10 + 8)
          || *a3 <= 0x15u && *a3 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)a3) )
        {
          v11 = sub_AD57C0(**(_QWORD **)(a1 + 24), **(unsigned __int8 ***)(a1 + 32), 0, 0);
          v13 = (unsigned __int8 *)v11;
          if ( *(_BYTE *)v11 == 17 )
          {
LABEL_16:
            LOBYTE(v14) = sub_B532C0(v11 + 24, &v20, 36);
            v6 = v14;
            goto LABEL_5;
          }
          v15 = *(_QWORD *)(v11 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
          {
            v11 = (__int64)sub_AD7630(v11, 0, v12);
            if ( v11 && *(_BYTE *)v11 == 17 )
              goto LABEL_16;
            if ( *(_BYTE *)(v15 + 8) == 17 )
            {
              v16 = *(_DWORD *)(v15 + 32);
              v6 = 0;
              if ( !v16 )
                goto LABEL_5;
              v17 = 0;
              while ( 1 )
              {
                v18 = (_BYTE *)sub_AD69F0(v13, v17);
                if ( !v18 )
                  break;
                if ( *v18 != 13 )
                {
                  if ( *v18 != 17 )
                    break;
                  LOBYTE(v19) = sub_B532C0((__int64)(v18 + 24), &v20, 36);
                  v6 = v19;
                  if ( !(_BYTE)v19 )
                    break;
                }
                if ( v16 == ++v17 )
                  goto LABEL_5;
              }
            }
          }
        }
      }
    }
  }
  v6 = 0;
LABEL_5:
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return v6;
}
