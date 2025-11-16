// Function: sub_30CCE90
// Address: 0x30cce90
//
__int64 __fastcall sub_30CCE90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  int v5; // r13d
  unsigned __int8 v6; // al
  unsigned __int8 **v7; // rdx
  unsigned int v8; // r13d
  unsigned __int8 v9; // al
  _BYTE **v10; // rdx
  _BYTE *v11; // rdi
  unsigned int v12; // r14d
  unsigned int v13; // eax
  unsigned __int8 *v14; // rax
  unsigned __int8 v15; // dl
  __int64 v16; // rdi
  size_t v17; // rdx
  __int8 *v18; // rsi
  __int64 v19; // r13
  unsigned __int8 v20; // al
  unsigned __int8 v22; // al
  unsigned __int8 **v23; // rdx
  unsigned __int8 *v24; // rax
  unsigned __int8 v25; // dl
  __int64 v26; // rdx
  __int64 v27; // [rsp+28h] [rbp-D8h]
  __int64 v28[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v30; // [rsp+50h] [rbp-B0h]
  __int64 v31; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v32[2]; // [rsp+80h] [rbp-80h] BYREF
  _QWORD v33[2]; // [rsp+90h] [rbp-70h] BYREF
  _QWORD *v34; // [rsp+A0h] [rbp-60h]
  _QWORD v35[10]; // [rsp+B0h] [rbp-50h] BYREF

  sub_B18290(a1, " at callsite ", 0xDu);
  v3 = sub_B10CD0(a2);
  if ( v3 )
  {
    v4 = v3;
    while ( 1 )
    {
      v27 = v4 - 16;
      v6 = *(_BYTE *)(v4 - 16);
      if ( (v6 & 2) != 0 )
        v7 = *(unsigned __int8 ***)(v4 - 32);
      else
        v7 = (unsigned __int8 **)(v27 - 8LL * ((v6 >> 2) & 0xF));
      v5 = *(_DWORD *)(v4 + 4);
      v8 = v5 - *((_DWORD *)sub_AF34D0(*v7) + 4);
      v9 = *(_BYTE *)(v4 - 16);
      if ( (v9 & 2) != 0 )
        v10 = *(_BYTE ***)(v4 - 32);
      else
        v10 = (_BYTE **)(v27 - 8LL * ((v9 >> 2) & 0xF));
      v11 = *v10;
      v12 = 0;
      if ( **v10 == 20 )
      {
        v13 = *((_DWORD *)v11 + 1);
        if ( (v13 & 7) == 7 && (v13 & 0xFFFFFFF8) != 0 )
        {
          if ( (v13 & 0x10000000) != 0 )
            v12 = HIWORD(v13) & 7;
          else
            v12 = (unsigned __int16)(v13 >> 3);
        }
        else
        {
          v12 = (unsigned __int8)v13;
          if ( !LOBYTE(qword_4F813A8[8]) )
          {
            v12 = 0;
            if ( (v13 & 1) == 0 )
            {
              v12 = (v13 >> 1) & 0x1F;
              if ( ((v13 >> 1) & 0x20) != 0 )
                v12 |= (v13 >> 2) & 0xFE0;
            }
          }
        }
      }
      v14 = sub_AF34D0(v11);
      v15 = *(v14 - 16);
      if ( (v15 & 2) != 0 )
      {
        v16 = *(_QWORD *)(*((_QWORD *)v14 - 4) + 24LL);
        if ( !v16 )
          goto LABEL_29;
      }
      else
      {
        v16 = *(_QWORD *)&v14[-8 * ((v15 >> 2) & 0xF) + 8];
        if ( !v16 )
          goto LABEL_29;
      }
      v18 = (__int8 *)sub_B91420(v16);
      if ( v17 )
        goto LABEL_15;
LABEL_29:
      v22 = *(_BYTE *)(v4 - 16);
      if ( (v22 & 2) != 0 )
        v23 = *(unsigned __int8 ***)(v4 - 32);
      else
        v23 = (unsigned __int8 **)(v27 - 8LL * ((v22 >> 2) & 0xF));
      v24 = sub_AF34D0(*v23);
      v25 = *(v24 - 16);
      if ( (v25 & 2) != 0 )
      {
        v18 = *(__int8 **)(*((_QWORD *)v24 - 4) + 16LL);
        if ( v18 )
          goto LABEL_33;
      }
      else
      {
        v18 = *(__int8 **)&v24[-8 * ((v25 >> 2) & 0xF)];
        if ( v18 )
        {
LABEL_33:
          v18 = (__int8 *)sub_B91420((__int64)v18);
          goto LABEL_15;
        }
      }
      v17 = 0;
LABEL_15:
      sub_B18290(a1, v18, v17);
      sub_B18290(a1, ":", 1u);
      sub_B169E0(v32, "Line", 4, v8);
      v19 = sub_23FD640(a1, (__int64)v32);
      sub_B18290(v19, ":", 1u);
      sub_B169E0(v28, "Column", 6, *(unsigned __int16 *)(v4 + 2));
      sub_23FD640(v19, (__int64)v28);
      if ( v30 != &v31 )
        j_j___libc_free_0((unsigned __int64)v30);
      if ( (__int64 *)v28[0] != &v29 )
        j_j___libc_free_0(v28[0]);
      if ( v34 != v35 )
        j_j___libc_free_0((unsigned __int64)v34);
      if ( (_QWORD *)v32[0] != v33 )
        j_j___libc_free_0(v32[0]);
      if ( v12 )
      {
        sub_B18290(a1, ".", 1u);
        sub_B169E0(v32, "Disc", 4, v12);
        sub_23FD640(a1, (__int64)v32);
        if ( v34 != v35 )
          j_j___libc_free_0((unsigned __int64)v34);
        if ( (_QWORD *)v32[0] != v33 )
          j_j___libc_free_0(v32[0]);
      }
      v20 = *(_BYTE *)(v4 - 16);
      if ( (v20 & 2) != 0 )
      {
        if ( *(_DWORD *)(v4 - 24) != 2 )
          return sub_B18290(a1, ";", 1u);
        v26 = *(_QWORD *)(v4 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v4 - 16) >> 6) & 0xF) != 2 )
          return sub_B18290(a1, ";", 1u);
        v26 = v27 - 8LL * ((v20 >> 2) & 0xF);
      }
      v4 = *(_QWORD *)(v26 + 8);
      if ( !v4 )
        return sub_B18290(a1, ";", 1u);
      sub_B18290(a1, " @ ", 3u);
    }
  }
  return sub_B18290(a1, ";", 1u);
}
