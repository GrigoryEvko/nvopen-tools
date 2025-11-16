// Function: sub_11E5BF0
// Address: 0x11e5bf0
//
__int64 __fastcall sub_11E5BF0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int8 *v8; // rbx
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v10; // rsi
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rax
  _QWORD **v15; // rdx
  int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rsi
  unsigned int *v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // [rsp+8h] [rbp-A8h]
  __int64 v22; // [rsp+18h] [rbp-98h]
  char v23[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v24; // [rsp+40h] [rbp-70h]
  _QWORD v25[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v26; // [rsp+70h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4);
  v24 = 257;
  v5 = *(_QWORD *)(a2 - 32LL * (v4 & 0x7FFFFFF));
  v6 = sub_AD6530(*(_QWORD *)(v5 + 8), a2);
  v7 = *(_QWORD *)(a3 + 80);
  v8 = (unsigned __int8 *)v6;
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v7 + 56LL);
  if ( v9 != sub_928890 )
  {
    v10 = 40;
    v11 = v9(v7, 40u, (_BYTE *)v5, v8);
LABEL_5:
    if ( v11 )
      goto LABEL_6;
    goto LABEL_7;
  }
  if ( *(_BYTE *)v5 <= 0x15u && *v8 <= 0x15u )
  {
    v10 = v5;
    v11 = sub_AAB310(0x28u, (unsigned __int8 *)v5, v8);
    goto LABEL_5;
  }
LABEL_7:
  v26 = 257;
  v11 = (__int64)sub_BD2C40(72, unk_3F10FD0);
  if ( v11 )
  {
    v15 = *(_QWORD ***)(v5 + 8);
    v16 = *((unsigned __int8 *)v15 + 8);
    if ( (unsigned int)(v16 - 17) > 1 )
    {
      v18 = sub_BCB2A0(*v15);
    }
    else
    {
      BYTE4(v22) = (_BYTE)v16 == 18;
      LODWORD(v22) = *((_DWORD *)v15 + 8);
      v17 = (__int64 *)sub_BCB2A0(*v15);
      v18 = sub_BCE1B0(v17, v22);
    }
    sub_B523C0(v11, v18, 53, 40, v5, (__int64)v8, (__int64)v25, 0, 0, 0);
  }
  v10 = v11;
  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v11,
    v23,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v19 = *(unsigned int **)a3;
  v21 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v21 )
  {
    do
    {
      v20 = *((_QWORD *)v19 + 1);
      v10 = *v19;
      v19 += 4;
      sub_B99FD0(v11, v10, v20);
    }
    while ( (unsigned int *)v21 != v19 );
  }
LABEL_6:
  v25[0] = "neg";
  v26 = 259;
  v12 = (_BYTE *)sub_AD6530(*(_QWORD *)(v5 + 8), v10);
  v13 = sub_929DE0((unsigned int **)a3, v12, (_BYTE *)v5, (__int64)v25, 0, 1);
  v26 = 257;
  return sub_B36550((unsigned int **)a3, v11, v13, v5, (__int64)v25, 0);
}
