// Function: sub_1112240
// Address: 0x1112240
//
_QWORD *__fastcall sub_1112240(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  const char *v8; // rax
  __int64 v9; // rdx
  unsigned __int8 *v10; // r10
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  _QWORD *v15; // r12
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // r10
  __int64 v19; // rdx
  int v20; // ecx
  int v21; // eax
  _QWORD *v22; // rdi
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // [rsp+0h] [rbp-B0h]
  unsigned __int8 *v31; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v32; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v33; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+18h] [rbp-98h]
  _QWORD v35[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v36; // [rsp+40h] [rbp-70h]
  unsigned int v37[8]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v38; // [rsp+70h] [rbp-40h]

  v7 = *a1;
  v8 = sub_BD5D20(a1[1]);
  v36 = 261;
  v35[0] = v8;
  v35[1] = v9;
  if ( a2 > 0xF )
  {
    v10 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v7 + 80) + 56LL))(
                               *(_QWORD *)(v7 + 80),
                               a2,
                               a3,
                               a4);
    if ( !v10 )
    {
      v38 = 257;
      v17 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
      v18 = v17;
      if ( v17 )
      {
        v19 = *(_QWORD *)(a3 + 8);
        v29 = (__int64)v17;
        v20 = *(unsigned __int8 *)(v19 + 8);
        if ( (unsigned int)(v20 - 17) > 1 )
        {
          v24 = sub_BCB2A0(*(_QWORD **)v19);
        }
        else
        {
          v21 = *(_DWORD *)(v19 + 32);
          v22 = *(_QWORD **)v19;
          BYTE4(v34) = (_BYTE)v20 == 18;
          LODWORD(v34) = v21;
          v23 = (__int64 *)sub_BCB2A0(v22);
          v24 = sub_BCE1B0(v23, v34);
        }
        sub_B523C0(v29, v24, 53, a2, a3, a4, (__int64)v37, 0, 0, 0);
        v18 = (unsigned __int8 *)v29;
      }
      v31 = v18;
      (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
        *(_QWORD *)(v7 + 88),
        v18,
        v35,
        *(_QWORD *)(v7 + 56),
        *(_QWORD *)(v7 + 64));
      v25 = *(_QWORD *)v7;
      v10 = v31;
      v26 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
      if ( *(_QWORD *)v7 != v26 )
      {
        do
        {
          v27 = *(_QWORD *)(v25 + 8);
          v28 = *(_DWORD *)v25;
          v25 += 16;
          v32 = v10;
          sub_B99FD0((__int64)v10, v28, v27);
          v10 = v32;
        }
        while ( v26 != v25 );
      }
    }
  }
  else
  {
    v37[1] = 0;
    v10 = (unsigned __int8 *)sub_B35C90(v7, a2, a3, a4, (__int64)v35, 0, v37[0], 0);
  }
  v33 = v10;
  if ( *v10 > 0x1Cu )
    sub_B45260(v10, a1[1], 1);
  v11 = 0;
  v12 = (__int64 *)sub_B43CA0(a1[1]);
  *(_QWORD *)v37 = *((_QWORD *)v33 + 1);
  v13 = sub_B6E160(v12, 0x192u, (__int64)v37, 1);
  v38 = 257;
  v14 = v13;
  if ( v13 )
    v11 = *(_QWORD *)(v13 + 24);
  v15 = sub_BD2CC0(88, 2u);
  if ( v15 )
  {
    sub_B44260((__int64)v15, **(_QWORD **)(v11 + 16), 56, 2u, 0, 0);
    v15[9] = 0;
    sub_B4A290((__int64)v15, v11, v14, (__int64 *)&v33, 1, (__int64)v37, 0, 0);
  }
  return v15;
}
