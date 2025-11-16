// Function: sub_2A6B3D0
// Address: 0x2a6b3d0
//
void __fastcall sub_2A6B3D0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // r15
  __int64 v4; // rax
  char v5; // al
  int v6; // edx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 **v14; // rbx
  char v15; // al
  unsigned __int8 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rdx
  unsigned __int64 v20; // r8
  _BYTE *v21; // rdi
  __int64 v22; // rdx
  __int64 *v23; // rax
  unsigned __int8 *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-C8h]
  unsigned __int8 v26[48]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 *v27; // [rsp+50h] [rbp-80h] BYREF
  __int64 v28; // [rsp+58h] [rbp-78h]
  _BYTE v29[112]; // [rsp+60h] [rbp-70h] BYREF

  v3 = *((_QWORD *)a2 - 4);
  v4 = *((_QWORD *)a2 + 1);
  if ( !v3 || *(_BYTE *)v3 )
  {
    v5 = *(_BYTE *)(v4 + 8);
    if ( v5 != 7 )
      goto LABEL_5;
  }
  else
  {
    v5 = *(_BYTE *)(v4 + 8);
    if ( *((_QWORD *)a2 + 10) != *(_QWORD *)(v3 + 24) )
    {
      if ( v5 == 7 )
        return;
LABEL_5:
      if ( v5 != 15 )
      {
LABEL_6:
        sub_2A62B80(&v27, a2);
        sub_2A689D0(a1, (__int64)a2, (unsigned __int8 *)&v27, 0x100000000LL);
        sub_22C0090((unsigned __int8 *)&v27);
        return;
      }
LABEL_34:
      sub_2A6A450(a1, (__int64)a2);
      return;
    }
    if ( v5 != 7 )
    {
      if ( v5 == 15 )
        goto LABEL_34;
      if ( !sub_B2FC80(*((_QWORD *)a2 - 4)) || !sub_971E80((__int64)a2, v3) )
        goto LABEL_6;
      v6 = *a2;
      v27 = (__int64 *)v29;
      v28 = 0x800000000LL;
      if ( v6 == 40 )
      {
        v7 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
      }
      else
      {
        v7 = -32;
        if ( v6 != 85 )
        {
          v7 = -96;
          if ( v6 != 34 )
LABEL_51:
            BUG();
        }
      }
      if ( (a2[7] & 0x80u) != 0 )
      {
        v8 = sub_BD2BC0((__int64)a2);
        v10 = v8 + v9;
        if ( (a2[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v10 >> 4) )
            goto LABEL_51;
        }
        else if ( (unsigned int)((v10 - sub_BD2BC0((__int64)a2)) >> 4) )
        {
          if ( (a2[7] & 0x80u) == 0 )
            goto LABEL_51;
          v11 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
          if ( (a2[7] & 0x80u) == 0 )
            BUG();
          v12 = sub_BD2BC0((__int64)a2);
          v7 -= 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
        }
      }
      v14 = (unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      if ( v14 == (unsigned __int8 **)&a2[v7] )
      {
LABEL_41:
        v21 = sub_2A68BC0(a1, a2);
        if ( !(unsigned __int8)sub_2A62E90(v21) )
        {
          if ( !*(_QWORD *)(a1 + 24) )
            sub_4263D6(v21, a2, v22);
          v23 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(a1 + 32))(a1 + 8, v3);
          v24 = (unsigned __int8 *)sub_97A150((__int64)a2, v3, v27, (unsigned int)v28, v23, 1);
          if ( !v24 )
          {
            if ( v27 != (__int64 *)v29 )
              _libc_free((unsigned __int64)v27);
            goto LABEL_6;
          }
          sub_2A68820(a1, (__int64)a2, v24);
          goto LABEL_37;
        }
      }
      else
      {
        while ( 1 )
        {
          v15 = *(_BYTE *)(*((_QWORD *)*v14 + 1) + 8LL);
          if ( v15 == 15 )
            break;
          if ( v15 != 9 )
          {
            v16 = (unsigned __int8 *)sub_2A68BC0(a1, *v14);
            sub_22C05A0((__int64)v26, v16);
            if ( v26[0] <= 1u )
              goto LABEL_40;
            if ( (unsigned __int8)sub_2A62E90(v26) )
            {
              sub_2A6A450(a1, (__int64)a2);
LABEL_40:
              sub_22C0090(v26);
              goto LABEL_37;
            }
            v17 = sub_2A637C0(a1, (__int64)v26, *((_QWORD *)*v14 + 1));
            v19 = (unsigned int)v28;
            v20 = (unsigned int)v28 + 1LL;
            if ( v20 > HIDWORD(v28) )
            {
              v25 = v17;
              sub_C8D5F0((__int64)&v27, v29, (unsigned int)v28 + 1LL, 8u, v20, v18);
              v19 = (unsigned int)v28;
              v17 = v25;
            }
            v27[v19] = v17;
            LODWORD(v28) = v28 + 1;
            sub_22C0090(v26);
          }
          v14 += 4;
          if ( &a2[v7] == (unsigned __int8 *)v14 )
            goto LABEL_41;
        }
      }
      sub_2A6A450(a1, (__int64)a2);
LABEL_37:
      if ( v27 != (__int64 *)v29 )
        _libc_free((unsigned __int64)v27);
    }
  }
}
