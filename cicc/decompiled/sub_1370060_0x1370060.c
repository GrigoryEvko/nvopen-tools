// Function: sub_1370060
// Address: 0x1370060
//
int __fastcall sub_1370060(__int64 *a1, const void *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v6; // rdi
  size_t v7; // rdx
  const void *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v17; // [rsp+0h] [rbp-30h]
  __int64 v18; // [rsp+8h] [rbp-28h]

  v4 = (__int64)a2;
  v6 = *a1;
  if ( !v6 )
  {
    v17 = a4;
    v18 = a3;
    v13 = sub_22077B0(192);
    a3 = v18;
    a4 = v17;
    v6 = v13;
    if ( v13 )
    {
      *(_QWORD *)(v13 + 8) = 0;
      v14 = v13 + 40;
      *(_QWORD *)(v14 - 24) = 0;
      *(_QWORD *)(v14 - 16) = 0;
      *(_QWORD *)(v6 + 48) = v14;
      *(_QWORD *)(v6 + 40) = v14;
      *(_QWORD *)(v6 + 32) = v14;
      *(_QWORD *)(v6 + 96) = v6 + 88;
      *(_QWORD *)(v6 + 88) = v6 + 88;
      *(_QWORD *)(v6 + 56) = 0;
      *(_QWORD *)(v6 + 64) = 0;
      *(_QWORD *)(v6 + 72) = 0;
      *(_QWORD *)(v6 + 80) = 0;
      *(_QWORD *)v6 = &unk_49E8978;
      *(_QWORD *)(v6 + 104) = 0;
      *(_QWORD *)(v6 + 112) = 0;
      *(_QWORD *)(v6 + 120) = 0;
      *(_QWORD *)(v6 + 128) = 0;
      *(_QWORD *)(v6 + 136) = 0;
      *(_QWORD *)(v6 + 144) = 0;
      *(_QWORD *)(v6 + 152) = 0;
      *(_QWORD *)(v6 + 160) = 0;
      *(_QWORD *)(v6 + 168) = 0;
      *(_QWORD *)(v6 + 176) = 0;
      *(_DWORD *)(v6 + 184) = 0;
    }
    v15 = *a1;
    *a1 = v6;
    if ( v15 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
      v6 = *a1;
      a4 = v17;
      a3 = v18;
    }
  }
  sub_136F890(v6, (__int64)a2, a3, a4);
  LODWORD(v8) = dword_4F98560;
  if ( dword_4F98560 )
  {
    if ( !qword_4F983A0[21]
      || (v6 = (__int64)a2, v8 = (const void *)sub_1649960(a2), v12 = v11, v7 = qword_4F983A0[21], v7 == v12)
      && (!v7 || (a2 = (const void *)qword_4F983A0[20], v6 = (__int64)v8, LODWORD(v8) = memcmp(v8, a2, v7), !(_DWORD)v8)) )
    {
      v6 = (__int64)a1;
      LODWORD(v8) = (unsigned int)sub_136FFE0(a1);
    }
  }
  if ( byte_4F98020 )
  {
    if ( !qword_4F97E60[21]
      || (v6 = v4, v8 = (const void *)sub_1649960(v4), v10 = v9, v7 = qword_4F97E60[21], v7 == v10)
      && (!v7 || (a2 = (const void *)qword_4F97E60[20], v6 = (__int64)v8, LODWORD(v8) = memcmp(v8, a2, v7), !(_DWORD)v8)) )
    {
      sub_16BA580(v6, a2, v7);
      LODWORD(v8) = sub_1368E20(a1);
    }
  }
  return (int)v8;
}
