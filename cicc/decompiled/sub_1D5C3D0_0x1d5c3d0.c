// Function: sub_1D5C3D0
// Address: 0x1d5c3d0
//
__int64 __fastcall sub_1D5C3D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // r14d
  __int64 (*v12)(); // rax
  unsigned int v13; // r13d
  __int64 v14; // rbx
  __int64 v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 v24; // rax
  __int64 v26; // [rsp+0h] [rbp-60h]
  _QWORD *v27; // [rsp+8h] [rbp-58h]
  __int64 v28[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v29; // [rsp+20h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 8);
  if ( !v3 || *(_QWORD *)(v3 + 8) || *(_QWORD *)(a1 + 40) != sub_1648700(*(_QWORD *)(a1 + 8))[5] )
  {
    v4 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
       ? *(_QWORD **)(a1 - 8)
       : (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v5 = *v4;
    if ( *(_BYTE *)(*v4 + 16LL) == 13
      || (v6 = v4[3], *(_BYTE *)(v6 + 16) == 13)
      || (v7 = *(_QWORD *)(v5 + 8)) == 0
      || *(_QWORD *)(v7 + 8)
      || (v24 = *(_QWORD *)(v6 + 8)) == 0
      || *(_QWORD *)(v24 + 8) )
    {
      for ( ; v3; v3 = *(_QWORD *)(v3 + 8) )
      {
        v8 = sub_1648700(v3);
        if ( *((_BYTE *)v8 + 16) != 75 )
          return 0;
        v9 = (*((_BYTE *)v8 + 23) & 0x40) != 0 ? (_QWORD *)*(v8 - 1) : &v8[-3 * (*((_DWORD *)v8 + 5) & 0xFFFFFFF)];
        v10 = v9[3];
        if ( *(_BYTE *)(v10 + 16) != 13 )
          return 0;
        v11 = *(_DWORD *)(v10 + 32);
        if ( v11 <= 0x40 )
        {
          if ( *(_QWORD *)(v10 + 24) )
            return 0;
        }
        else if ( v11 != (unsigned int)sub_16A57B0(v10 + 24) )
        {
          return 0;
        }
      }
      v12 = *(__int64 (**)())(*(_QWORD *)a2 + 192LL);
      if ( v12 != sub_1D5A3A0 )
      {
        v13 = ((__int64 (__fastcall *)(__int64, __int64))v12)(a2, a1);
        if ( (_BYTE)v13 )
        {
          v14 = *(_QWORD *)(a1 + 8);
          if ( !v14 )
          {
LABEL_44:
            sub_15F20C0((_QWORD *)a1);
            return v13;
          }
          while ( 1 )
          {
            v16 = sub_1648700(v14);
            v27 = (_QWORD *)v14;
            v14 = *(_QWORD *)(v14 + 8);
            v17 = (__int64)v16;
            if ( v16[5] == *(_QWORD *)(a1 + 40) )
              v17 = a1;
            v29 = 257;
            v18 = sub_13CF970(a1);
            v19 = sub_15FB440(26, *(__int64 **)v18, *(_QWORD *)(v18 + 24), (__int64)v28, v17);
            v20 = *(_QWORD *)(a1 + 48);
            v21 = v19;
            v28[0] = v20;
            if ( v20 )
              break;
            v15 = v19 + 48;
            if ( (__int64 *)(v19 + 48) != v28 )
            {
              v22 = *(_QWORD *)(v19 + 48);
              if ( v22 )
                goto LABEL_31;
            }
LABEL_25:
            sub_1593B40(v27, v21);
            if ( !v14 )
              goto LABEL_44;
          }
          sub_1623A60((__int64)v28, v20, 2);
          v15 = v21 + 48;
          if ( (__int64 *)(v21 + 48) == v28 )
          {
            if ( v28[0] )
              sub_161E7C0((__int64)v28, v28[0]);
            goto LABEL_25;
          }
          v22 = *(_QWORD *)(v21 + 48);
          if ( v22 )
          {
LABEL_31:
            v26 = v15;
            sub_161E7C0(v15, v22);
            v15 = v26;
          }
          v23 = (unsigned __int8 *)v28[0];
          *(_QWORD *)(v21 + 48) = v28[0];
          if ( v23 )
            sub_1623210((__int64)v28, v23, v15);
          goto LABEL_25;
        }
      }
    }
  }
  return 0;
}
