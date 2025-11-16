// Function: sub_F292F0
// Address: 0xf292f0
//
__int64 __fastcall sub_F292F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v5; // rax
  unsigned __int8 **v6; // r13
  unsigned __int8 **v7; // r14
  unsigned __int8 **v8; // r15
  __int64 *v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 i; // r14
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // r13
  unsigned __int8 *v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // r14
  __int64 v23; // rax
  _QWORD *v24; // [rsp+8h] [rbp-98h]
  _QWORD *v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+8h] [rbp-98h]
  _QWORD v27[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v28; // [rsp+30h] [rbp-70h]
  __int64 v29[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v30; // [rsp+60h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 <= 0x1Cu )
    return 0;
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 || *(_QWORD *)(v3 + 8) || *(_BYTE *)v2 == 84 || sub_98CD60((unsigned __int8 *)v2, 0) )
    return 0;
  v5 = 4LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
  {
    v6 = *(unsigned __int8 ***)(v2 - 8);
    v7 = &v6[v5];
  }
  else
  {
    v7 = (unsigned __int8 **)v2;
    v6 = (unsigned __int8 **)(v2 - v5 * 8);
  }
  if ( v7 == v6 )
  {
    sub_B44F30((unsigned __int8 *)v2);
    sub_B44B50((__int64 *)v2, 0);
    sub_B44A60(v2);
  }
  else
  {
    v8 = 0;
    do
    {
      if ( **v6 != 24 && !sub_98ED60(*v6, 0, 0, 0, 0) )
      {
        if ( v8 )
          return 0;
        v8 = v6;
      }
      v6 += 4;
    }
    while ( v7 != v6 );
    sub_B44F30((unsigned __int8 *)v2);
    sub_B44B50((__int64 *)v2, 0);
    sub_B44A60(v2);
    if ( v8 )
    {
      sub_D5F1F0(*(_QWORD *)(a1 + 32), v2);
      v9 = *(__int64 **)(a1 + 32);
      v27[0] = sub_BD5D20((__int64)*v8);
      v27[2] = ".fr";
      v28 = 773;
      v27[1] = v10;
      v11 = (__int64)*v8;
      v30 = 257;
      v12 = sub_BD2C40(72, unk_3F10A14);
      if ( v12 )
      {
        v24 = v12;
        sub_B549F0((__int64)v12, v11, (__int64)v29, 0, 0);
        v12 = v24;
      }
      v25 = v12;
      (*(void (__fastcall **)(__int64, _QWORD *, _QWORD *, __int64, __int64))(*(_QWORD *)v9[11] + 16LL))(
        v9[11],
        v12,
        v27,
        v9[7],
        v9[8]);
      v13 = *v9;
      v14 = (__int64)v25;
      for ( i = *v9 + 16LL * *((unsigned int *)v9 + 2); i != v13; v14 = v26 )
      {
        v16 = *(_QWORD *)(v13 + 8);
        v17 = *(_DWORD *)v13;
        v13 += 16;
        v26 = v14;
        sub_B99FD0(v14, v17, v16);
      }
      v18 = (__int64)*v8;
      if ( *v8 )
      {
        v19 = v8[1];
        *(_QWORD *)v8[2] = v19;
        if ( v19 )
          *((_QWORD *)v19 + 2) = v8[2];
      }
      *v8 = (unsigned __int8 *)v14;
      if ( v14 )
      {
        v20 = *(_QWORD *)(v14 + 16);
        v8[1] = (unsigned __int8 *)v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = v8 + 1;
        v8[2] = (unsigned __int8 *)(v14 + 16);
        *(_QWORD *)(v14 + 16) = v8;
      }
      if ( *(_BYTE *)v18 > 0x1Cu )
      {
        v21 = *(_QWORD *)(a1 + 40);
        v29[0] = v18;
        v22 = v21 + 2096;
        sub_F200C0(v22, v29);
        v23 = *(_QWORD *)(v18 + 16);
        if ( v23 )
        {
          if ( !*(_QWORD *)(v23 + 8) )
          {
            v29[0] = *(_QWORD *)(v23 + 24);
            sub_F200C0(v22, v29);
          }
        }
      }
    }
  }
  return v2;
}
