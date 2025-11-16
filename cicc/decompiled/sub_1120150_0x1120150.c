// Function: sub_1120150
// Address: 0x1120150
//
unsigned __int8 *__fastcall sub_1120150(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v8; // ebx
  bool v9; // al
  unsigned int v10; // r14d
  unsigned int v11; // edx
  int v12; // r9d
  __int64 v13; // rax
  _QWORD *v15; // r12
  unsigned int v17; // r9d
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  const void *v21; // rax
  __int16 v22; // r14
  __int64 v23; // rbx
  _QWORD **v24; // rdx
  int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // rsi
  int v28; // ecx
  __int64 *v29; // rax
  __int64 v32; // rsi
  int v33; // ecx
  __int64 *v34; // rax
  bool v35; // al
  int v36; // [rsp+Ch] [rbp-94h]
  unsigned int v37; // [rsp+Ch] [rbp-94h]
  __int64 v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+28h] [rbp-78h]
  __int64 v40; // [rsp+30h] [rbp-70h]
  __int64 v41; // [rsp+38h] [rbp-68h]
  const void *v42; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v43; // [rsp+48h] [rbp-58h]
  __int16 v44; // [rsp+60h] [rbp-40h]

  v8 = *(_DWORD *)(a5 + 8);
  v38 = a2;
  if ( v8 <= 0x40 )
    v9 = *(_QWORD *)a5 == 0;
  else
    v9 = v8 == (unsigned int)sub_C444A0(a5);
  if ( !v9 )
  {
    if ( v8 <= 0x40 )
    {
      _RDX = *(const void **)a5;
      v10 = 64;
      __asm { tzcnt   rax, rdx }
      if ( *(_QWORD *)a5 )
        v10 = _RAX;
      if ( v8 <= v10 )
        v10 = v8;
    }
    else
    {
      v10 = sub_C44590(a5);
    }
    v11 = *(_DWORD *)(a4 + 8);
    if ( v11 <= 0x40 )
    {
      _RAX = *(const void **)a4;
      if ( *(_QWORD *)a4 || !v10 )
      {
        if ( _RAX != *(const void **)a5 )
        {
          v17 = 64;
          __asm { tzcnt   rcx, rax }
          if ( _RAX )
            v17 = _RCX;
          if ( v11 <= v17 )
            v17 = *(_DWORD *)(a4 + 8);
          v12 = v17 - v10;
          if ( v12 <= 0 )
            goto LABEL_11;
          goto LABEL_23;
        }
LABEL_34:
        v22 = 32;
        v23 = sub_AD6530(*(_QWORD *)(a3 + 8), a2);
        if ( (*(_WORD *)(v38 + 2) & 0x3F) == 0x21 )
          v22 = sub_B52870(32);
        v44 = 257;
        v15 = sub_BD2C40(72, unk_3F10FD0);
        if ( !v15 )
          return (unsigned __int8 *)v15;
        v24 = *(_QWORD ***)(a3 + 8);
        v28 = *((unsigned __int8 *)v24 + 8);
        if ( (unsigned int)(v28 - 17) <= 1 )
        {
          BYTE4(v40) = (_BYTE)v28 == 18;
          LODWORD(v40) = *((_DWORD *)v24 + 8);
          v29 = (__int64 *)sub_BCB2A0(*v24);
          v27 = sub_BCE1B0(v29, v40);
          goto LABEL_40;
        }
LABEL_39:
        v27 = sub_BCB2A0(*v24);
        goto LABEL_40;
      }
    }
    else
    {
      v36 = *(_DWORD *)(a4 + 8);
      if ( v36 != (unsigned int)sub_C444A0(a4) || !v10 )
      {
        a2 = a5;
        if ( !sub_C43C50(a4, (const void **)a5) )
        {
          v12 = sub_C44590(a4) - v10;
          if ( v12 <= 0 )
          {
LABEL_11:
            v13 = sub_AD64C0(*(_QWORD *)(v38 + 8), (*(_WORD *)(v38 + 2) & 0x3F) == 33, 0);
            return sub_F162A0(a1, v38, v13);
          }
LABEL_23:
          v43 = v8;
          if ( v8 > 0x40 )
          {
            v37 = v12;
            sub_C43780((__int64)&v42, (const void **)a5);
            v8 = v43;
            v12 = v37;
            if ( v43 > 0x40 )
            {
              sub_C47690((__int64 *)&v42, v37);
              v12 = v37;
              if ( v43 <= 0x40 )
              {
                if ( *(const void **)a4 != v42 )
                  goto LABEL_11;
              }
              else
              {
                v35 = sub_C43C50((__int64)&v42, (const void **)a4);
                v12 = v37;
                if ( !v35 )
                {
                  if ( v42 )
                    j_j___libc_free_0_0(v42);
                  goto LABEL_11;
                }
                if ( v42 )
                {
                  j_j___libc_free_0_0(v42);
                  v12 = v37;
                }
              }
              goto LABEL_29;
            }
          }
          else
          {
            v42 = *(const void **)a5;
          }
          if ( v12 == v8 )
          {
            v19 = 0;
            v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
          }
          else
          {
            v19 = (_QWORD)v42 << v12;
            v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
            if ( !v8 )
            {
              v21 = 0;
LABEL_28:
              v42 = v21;
              if ( v21 != *(const void **)a4 )
                goto LABEL_11;
LABEL_29:
              v22 = 32;
              v23 = sub_AD64C0(*(_QWORD *)(a3 + 8), v12, 0);
              if ( (*(_WORD *)(v38 + 2) & 0x3F) == 0x21 )
                v22 = sub_B52870(32);
              v44 = 257;
              v15 = sub_BD2C40(72, unk_3F10FD0);
              if ( !v15 )
                return (unsigned __int8 *)v15;
              v24 = *(_QWORD ***)(a3 + 8);
              v25 = *((unsigned __int8 *)v24 + 8);
              if ( (unsigned int)(v25 - 17) <= 1 )
              {
                BYTE4(v41) = (_BYTE)v25 == 18;
                LODWORD(v41) = *((_DWORD *)v24 + 8);
                v26 = (__int64 *)sub_BCB2A0(*v24);
                v27 = sub_BCE1B0(v26, v41);
LABEL_40:
                sub_B523C0((__int64)v15, v27, 53, v22, a3, v23, (__int64)&v42, 0, 0, 0);
                return (unsigned __int8 *)v15;
              }
              goto LABEL_39;
            }
          }
          v21 = (const void *)(v19 & v20);
          goto LABEL_28;
        }
        goto LABEL_34;
      }
    }
    v32 = v8 - v10;
    v22 = 35;
    v23 = sub_AD64C0(*(_QWORD *)(a3 + 8), v32, 0);
    if ( (*(_WORD *)(v38 + 2) & 0x3F) == 0x21 )
      v22 = sub_B52870(35);
    v44 = 257;
    v15 = sub_BD2C40(72, unk_3F10FD0);
    if ( !v15 )
      return (unsigned __int8 *)v15;
    v24 = *(_QWORD ***)(a3 + 8);
    v33 = *((unsigned __int8 *)v24 + 8);
    if ( (unsigned int)(v33 - 17) <= 1 )
    {
      BYTE4(v39) = (_BYTE)v33 == 18;
      LODWORD(v39) = *((_DWORD *)v24 + 8);
      v34 = (__int64 *)sub_BCB2A0(*v24);
      v27 = sub_BCE1B0(v34, v39);
      goto LABEL_40;
    }
    goto LABEL_39;
  }
  return 0;
}
