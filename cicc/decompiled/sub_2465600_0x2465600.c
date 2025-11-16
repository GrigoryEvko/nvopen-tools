// Function: sub_2465600
// Address: 0x2465600
//
__int64 __fastcall sub_2465600(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 i; // rdi
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int8 *v13; // r15
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  _QWORD **v16; // rdx
  int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // [rsp+8h] [rbp-68h]
  char v26[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v27; // [rsp+30h] [rbp-40h]

  v6 = a2;
  for ( i = *(_QWORD *)(a2 + 8); *(_BYTE *)(i + 8) != 12; v6 = v9 )
  {
    v9 = sub_24650D0(a1, v6, a3);
    i = *(_QWORD *)(v9 + 8);
  }
  v10 = v6;
  if ( *(_DWORD *)(i + 8) >> 8 != 1 )
  {
    v11 = sub_AD64C0(i, 0, 0);
    v12 = *(_QWORD *)(a3 + 80);
    v13 = (unsigned __int8 *)v11;
    v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v12 + 56LL);
    if ( v14 == sub_928890 )
    {
      if ( *(_BYTE *)v6 > 0x15u || *v13 > 0x15u )
      {
LABEL_10:
        v27 = 257;
        v10 = (__int64)sub_BD2C40(72, unk_3F10FD0);
        if ( v10 )
        {
          v16 = *(_QWORD ***)(v6 + 8);
          v17 = *((unsigned __int8 *)v16 + 8);
          if ( (unsigned int)(v17 - 17) > 1 )
          {
            v19 = sub_BCB2A0(*v16);
          }
          else
          {
            BYTE4(v25) = (_BYTE)v17 == 18;
            LODWORD(v25) = *((_DWORD *)v16 + 8);
            v18 = (__int64 *)sub_BCB2A0(*v16);
            v19 = sub_BCE1B0(v18, v25);
          }
          sub_B523C0(v10, v19, 53, 33, v6, (__int64)v13, (__int64)v26, 0, 0, 0);
        }
        (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
          *(_QWORD *)(a3 + 88),
          v10,
          a4,
          *(_QWORD *)(a3 + 56),
          *(_QWORD *)(a3 + 64));
        v20 = 16LL * *(unsigned int *)(a3 + 8);
        v21 = *(_QWORD *)a3;
        v22 = v21 + v20;
        while ( v22 != v21 )
        {
          v23 = *(_QWORD *)(v21 + 8);
          v24 = *(_DWORD *)v21;
          v21 += 16;
          sub_B99FD0(v10, v24, v23);
        }
        return v10;
      }
      v10 = sub_AAB310(0x21u, (unsigned __int8 *)v6, v13);
    }
    else
    {
      v10 = v14(v12, 33u, (_BYTE *)v6, v13);
    }
    if ( v10 )
      return v10;
    goto LABEL_10;
  }
  return v10;
}
