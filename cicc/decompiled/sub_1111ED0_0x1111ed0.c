// Function: sub_1111ED0
// Address: 0x1111ed0
//
__int64 __fastcall sub_1111ED0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // r15d
  _QWORD *v6; // r13
  __int64 *v7; // r14
  _BYTE *v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // r15
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned int v18; // esi
  _QWORD **v19; // rdx
  int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned int v26; // esi
  unsigned int v27; // [rsp+Ch] [rbp-B4h]
  __int64 v28; // [rsp+10h] [rbp-B0h]
  __int64 v29; // [rsp+10h] [rbp-B0h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+28h] [rbp-98h]
  _BYTE v32[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v33; // [rsp+50h] [rbp-70h]
  _BYTE v34[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v35; // [rsp+80h] [rbp-40h]

  v2 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
  v3 = *(_QWORD *)(a1 + 16);
  v33 = 257;
  v28 = sub_AD6530(*(_QWORD *)(*(_QWORD *)v3 + 8LL), a2);
  v4 = **(_QWORD **)(a1 + 16);
  v5 = **(_DWORD **)(a1 + 8);
  v6 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v2[10] + 56LL))(
                   v2[10],
                   v5,
                   v4,
                   v28);
  if ( !v6 )
  {
    v35 = 257;
    v6 = sub_BD2C40(72, unk_3F10FD0);
    if ( v6 )
    {
      v12 = *(_QWORD *)(v4 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 > 1 )
      {
        v14 = sub_BCB2A0(*(_QWORD **)v12);
      }
      else
      {
        BYTE4(v30) = *(_BYTE *)(v12 + 8) == 18;
        LODWORD(v30) = *(_DWORD *)(v12 + 32);
        v13 = (__int64 *)sub_BCB2A0(*(_QWORD **)v12);
        v14 = sub_BCE1B0(v13, v30);
      }
      sub_B523C0((__int64)v6, v14, 53, v5, v4, v28, (__int64)v34, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, _QWORD *, _BYTE *, __int64, __int64))(*(_QWORD *)v2[11] + 16LL))(
      v2[11],
      v6,
      v32,
      v2[7],
      v2[8]);
    v15 = *v2;
    v16 = *v2 + 16LL * *((unsigned int *)v2 + 2);
    while ( v16 != v15 )
    {
      v17 = *(_QWORD *)(v15 + 8);
      v18 = *(_DWORD *)v15;
      v15 += 16;
      sub_B99FD0((__int64)v6, v18, v17);
    }
  }
  v7 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
  v8 = *(_BYTE **)(a1 + 24);
  v33 = 257;
  v29 = sub_AD64C0(*(_QWORD *)(**(_QWORD **)(a1 + 16) + 8LL), *v8 == 0 ? 1LL : -1LL, 1u);
  v9 = **(_QWORD **)(a1 + 16);
  v27 = **(_DWORD **)(a1 + 8);
  v10 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v7[10] + 56LL))(
                    v7[10],
                    v27,
                    v9,
                    v29);
  if ( !v10 )
  {
    v35 = 257;
    v10 = sub_BD2C40(72, unk_3F10FD0);
    if ( v10 )
    {
      v19 = *(_QWORD ***)(v9 + 8);
      v20 = *((unsigned __int8 *)v19 + 8);
      if ( (unsigned int)(v20 - 17) > 1 )
      {
        v22 = sub_BCB2A0(*v19);
      }
      else
      {
        BYTE4(v31) = (_BYTE)v20 == 18;
        LODWORD(v31) = *((_DWORD *)v19 + 8);
        v21 = (__int64 *)sub_BCB2A0(*v19);
        v22 = sub_BCE1B0(v21, v31);
      }
      sub_B523C0((__int64)v10, v22, 53, v27, v9, v29, (__int64)v34, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, _QWORD *, _BYTE *, __int64, __int64))(*(_QWORD *)v7[11] + 16LL))(
      v7[11],
      v10,
      v32,
      v7[7],
      v7[8]);
    v23 = *v7;
    v24 = *v7 + 16LL * *((unsigned int *)v7 + 2);
    while ( v24 != v23 )
    {
      v25 = *(_QWORD *)(v23 + 8);
      v26 = *(_DWORD *)v23;
      v23 += 16;
      sub_B99FD0((__int64)v10, v26, v25);
    }
  }
  v35 = 257;
  return sub_B504D0((unsigned int)(**(_DWORD **)(a1 + 8) == 32) + 28, (__int64)v6, (__int64)v10, (__int64)v34, 0, 0);
}
