// Function: sub_E018D0
// Address: 0xe018d0
//
__int64 __fastcall sub_E018D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 v7; // rax
  unsigned __int8 *v8; // r14
  unsigned __int8 v9; // al
  unsigned __int8 *v10; // r13
  unsigned __int8 v11; // al
  unsigned __int8 v12; // al
  int v13; // ecx
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int8 *v17; // rcx
  _BYTE *v18; // rdi
  int v19; // edx
  int v20; // r9d
  __int64 v21; // rsi
  unsigned int v22; // r12d
  unsigned __int8 **v23; // rax
  int v24; // edi
  unsigned int v25; // ecx
  unsigned __int8 **v26; // rax
  int v27; // edi
  int v28; // edi
  __int64 *v30; // [rsp+8h] [rbp-F8h]
  unsigned __int8 *v33; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v34; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v35; // [rsp+38h] [rbp-C8h]
  __int64 v36; // [rsp+40h] [rbp-C0h]
  __int64 v37; // [rsp+48h] [rbp-B8h]
  _BYTE *v38; // [rsp+50h] [rbp-B0h]
  __int64 v39; // [rsp+58h] [rbp-A8h]
  _BYTE v40[32]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int8 *v41; // [rsp+80h] [rbp-80h] BYREF
  __int64 v42; // [rsp+88h] [rbp-78h]
  __int64 v43; // [rsp+90h] [rbp-70h]
  __int64 v44; // [rsp+98h] [rbp-68h]
  _BYTE *v45; // [rsp+A0h] [rbp-60h]
  __int64 v46; // [rsp+A8h] [rbp-58h]
  _BYTE v47[80]; // [rsp+B0h] [rbp-50h] BYREF

  v6 = *(_BYTE *)(a2 - 16);
  v30 = (__int64 *)a3;
  if ( (v6 & 2) != 0 )
  {
    v7 = *(_QWORD *)(a2 - 32);
  }
  else
  {
    a3 = a2 - 8LL * ((v6 >> 2) & 0xF);
    v7 = a3 - 16;
  }
  v8 = *(unsigned __int8 **)(v7 + 8);
  if ( v8 && (unsigned __int8)(*v8 - 5) >= 0x20u )
    v8 = 0;
  v9 = *(_BYTE *)(a1 - 16);
  if ( (v9 & 2) != 0 )
  {
    v10 = *(unsigned __int8 **)(*(_QWORD *)(a1 - 32) + 8LL);
    if ( !v10 )
      goto LABEL_53;
  }
  else
  {
    a3 = a1 - 8LL * ((v9 >> 2) & 0xF);
    v10 = *(unsigned __int8 **)(a3 - 8);
    if ( !v10 )
      goto LABEL_53;
  }
  if ( (unsigned __int8)(*v10 - 5) <= 0x1Fu && v8 )
  {
    if ( v10 == v8 )
      goto LABEL_31;
    v34 = 0;
    v38 = v40;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v39 = 0x400000000LL;
    while ( 1 )
    {
      while ( 1 )
      {
        v41 = v10;
        if ( !(unsigned __int8)sub_E01260((__int64)&v34, (__int64 *)&v41, a3, a4, a5, a6) )
LABEL_64:
          sub_C64ED0("Cycle found in TBAA metadata.", 1u);
        v11 = *(v10 - 16);
        if ( (v11 & 2) != 0 )
          break;
        a4 = (*((_WORD *)v10 - 8) >> 6) & 0xF;
        if ( (unsigned int)a4 <= 2 )
        {
          if ( (_DWORD)a4 != 2 )
            goto LABEL_16;
          a4 = -16 - 8LL * ((v11 >> 2) & 0xF);
          v23 = (unsigned __int8 **)&v10[a4];
          goto LABEL_35;
        }
        v23 = (unsigned __int8 **)&v10[-16 - 8LL * ((v11 >> 2) & 0xF)];
        v10 = *v23;
        v24 = **v23;
        a4 = (unsigned int)(v24 - 5);
        if ( (unsigned __int8)(v24 - 5) > 0x1Fu )
          goto LABEL_35;
      }
      if ( *((_DWORD *)v10 - 6) <= 2u )
      {
        if ( *((_DWORD *)v10 - 6) != 2 )
          goto LABEL_16;
        v23 = (unsigned __int8 **)*((_QWORD *)v10 - 4);
        goto LABEL_35;
      }
      v23 = (unsigned __int8 **)*((_QWORD *)v10 - 4);
      v10 = *v23;
      v28 = **v23;
      a4 = (unsigned int)(v28 - 5);
      if ( (unsigned __int8)(v28 - 5) > 0x1Fu )
      {
LABEL_35:
        v10 = v23[1];
        if ( !v10 || (unsigned __int8)(*v10 - 5) > 0x1Fu )
        {
LABEL_16:
          v41 = 0;
          v42 = 0;
          v43 = 0;
          v44 = 0;
          v45 = v47;
          v46 = 0x400000000LL;
          while ( 1 )
          {
            while ( 1 )
            {
              v33 = v8;
              if ( !(unsigned __int8)sub_E01260((__int64)&v41, (__int64 *)&v33, a3, a4, a5, a6) )
                goto LABEL_64;
              v12 = *(v8 - 16);
              if ( (v12 & 2) != 0 )
                break;
              v25 = (*((_WORD *)v8 - 8) >> 6) & 0xF;
              if ( v25 <= 2 )
              {
                if ( v25 != 2 )
                  goto LABEL_21;
                a4 = -16 - 8LL * ((v12 >> 2) & 0xF);
                v26 = (unsigned __int8 **)&v8[a4];
                goto LABEL_40;
              }
              v26 = (unsigned __int8 **)&v8[-16 - 8LL * ((v12 >> 2) & 0xF)];
              v8 = *v26;
              v27 = **v26;
              a4 = (unsigned int)(v27 - 5);
              if ( (unsigned __int8)(v27 - 5) > 0x1Fu )
                goto LABEL_40;
            }
            if ( *((_DWORD *)v8 - 6) <= 2u )
            {
              if ( *((_DWORD *)v8 - 6) != 2 )
                goto LABEL_21;
              v26 = (unsigned __int8 **)*((_QWORD *)v8 - 4);
              goto LABEL_40;
            }
            v26 = (unsigned __int8 **)*((_QWORD *)v8 - 4);
            v8 = *v26;
            a3 = **v26;
            a4 = (unsigned int)(a3 - 5);
            if ( (unsigned __int8)(a3 - 5) > 0x1Fu )
            {
LABEL_40:
              v8 = v26[1];
              if ( !v8 || (unsigned __int8)(*v8 - 5) > 0x1Fu )
              {
LABEL_21:
                v13 = v39 - 1;
                v14 = (unsigned int)(v46 - 1);
                if ( (int)v46 - 1 < 0 || v13 < 0 )
                {
                  v8 = 0;
                  if ( v45 != v47 )
LABEL_27:
                    _libc_free(v45, v14);
                }
                else
                {
                  v15 = (int)v39 - 2;
                  v16 = -8 * v15 + 8LL * v13;
                  v17 = 0;
                  v18 = &v38[v16];
                  v14 = (unsigned __int64)&v45[8 * (int)v14 + -8 * v15];
                  do
                  {
                    v8 = v17;
                    v17 = *(unsigned __int8 **)&v18[8 * v15];
                    if ( v17 != *(unsigned __int8 **)(v14 + 8 * v15) )
                    {
                      if ( v45 == v47 )
                        goto LABEL_28;
                      goto LABEL_27;
                    }
                    v19 = v46 - v39 + v15;
                    v20 = v15--;
                  }
                  while ( v20 >= 0 && v19 >= 0 );
                  v8 = v17;
                  if ( v45 != v47 )
                    goto LABEL_27;
                }
LABEL_28:
                v21 = 8LL * (unsigned int)v44;
                sub_C7D6A0(v42, v21, 8);
                if ( v38 != v40 )
                  _libc_free(v38, v21);
                sub_C7D6A0(v35, 8LL * (unsigned int)v37, 8);
                if ( v8 )
                {
LABEL_31:
                  if ( (unsigned __int8)sub_DFF9C0(a1, a2, (__int64)v8, v30, &v41) )
                    return (unsigned __int8)v41;
                  v22 = sub_DFF9C0(a2, a1, (__int64)v8, v30, &v41);
                  if ( (_BYTE)v22 )
                  {
                    return (unsigned __int8)v41;
                  }
                  else if ( v30 )
                  {
                    *v30 = sub_DFF6F0((__int64)v8);
                  }
                  return v22;
                }
                goto LABEL_53;
              }
            }
          }
        }
      }
    }
  }
LABEL_53:
  v22 = 1;
  if ( v30 )
    *v30 = 0;
  return v22;
}
