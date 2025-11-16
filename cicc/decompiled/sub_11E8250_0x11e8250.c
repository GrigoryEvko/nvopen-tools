// Function: sub_11E8250
// Address: 0x11e8250
//
__int64 __fastcall sub_11E8250(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        __int64 a7)
{
  int v9; // r11d
  unsigned __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned int v13; // r11d
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  char v18; // al
  _QWORD *v19; // rax
  __int64 v20; // r12
  unsigned int *v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rdx
  unsigned int v24; // esi
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // [rsp+4h] [rbp-9Ch]
  __int64 v31; // [rsp+8h] [rbp-98h]
  __int64 v32; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  __int64 *v34; // [rsp+18h] [rbp-88h]
  char v36; // [rsp+20h] [rbp-80h]
  unsigned int v37; // [rsp+28h] [rbp-78h]
  unsigned int v38; // [rsp+28h] [rbp-78h]
  _QWORD **v39; // [rsp+28h] [rbp-78h]
  _BYTE *v40; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v41[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v42; // [rsp+60h] [rbp-40h]

  v9 = *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL);
  v10 = 0;
  if ( v9 )
    v10 = (1LL << ((unsigned __int8)v9 - 1)) - 1;
  v11 = 0;
  v37 = *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL);
  if ( a5 <= v10 )
  {
    v11 = sub_AD64C0(*(_QWORD *)(a2 + 8), a5, 0);
    if ( a6 )
    {
      v12 = a5 + 1;
      v13 = v37;
      if ( a6 <= a5 )
        v12 = a6 - 1;
      v32 = v12;
      v31 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      if ( v12 )
      {
        if ( a3 )
        {
          v30 = v37;
          v34 = *(__int64 **)(a1 + 24);
          v39 = (_QWORD **)sub_B43CA0(a2);
          v26 = sub_97FA80(*v34, (__int64)v39);
          v27 = sub_BCCE00(*v39, v26);
          v28 = sub_ACD640(v27, v32, 0);
          v29 = sub_B343C0(a7, 0xEEu, v31, 0x100u, a3, 0x100u, v28, 0, 0, 0, 0, 0);
          v13 = v30;
          if ( v29 )
          {
            if ( *(_BYTE *)v29 == 85 )
              *(_WORD *)(v29 + 2) = *(_WORD *)(v29 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
          }
        }
      }
      v38 = v13;
      if ( a6 <= a5 )
      {
        v14 = sub_BCB2B0(*(_QWORD **)(a7 + 72));
        v15 = sub_BCD140(*(_QWORD **)(a7 + 72), v38);
        v40 = (_BYTE *)sub_ACD640(v15, v32, 0);
        v41[0] = "endptr";
        v42 = 259;
        v33 = sub_921130((unsigned int **)a7, v14, v31, &v40, 1, (__int64)v41, 3u);
        v16 = sub_AD64C0(v14, 0, 0);
        v17 = sub_AA4E30(*(_QWORD *)(a7 + 48));
        v18 = sub_AE5020(v17, *(_QWORD *)(v16 + 8));
        v42 = 257;
        v36 = v18;
        v19 = sub_BD2C40(80, unk_3F10A10);
        v20 = (__int64)v19;
        if ( v19 )
          sub_B4D3C0((__int64)v19, v16, v33, 0, v36, v33, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a7 + 88) + 16LL))(
          *(_QWORD *)(a7 + 88),
          v20,
          v41,
          *(_QWORD *)(a7 + 56),
          *(_QWORD *)(a7 + 64));
        v21 = *(unsigned int **)a7;
        v22 = *(_QWORD *)a7 + 16LL * *(unsigned int *)(a7 + 8);
        if ( *(_QWORD *)a7 != v22 )
        {
          do
          {
            v23 = *((_QWORD *)v21 + 1);
            v24 = *v21;
            v21 += 4;
            sub_B99FD0(v20, v24, v23);
          }
          while ( (unsigned int *)v22 != v21 );
        }
      }
    }
  }
  return v11;
}
