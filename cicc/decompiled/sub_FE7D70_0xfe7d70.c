// Function: sub_FE7D70
// Address: 0xfe7d70
//
int __fastcall sub_FE7D70(__int64 *a1, const char *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v6; // rdi
  const char *v7; // rax
  size_t v8; // r14
  size_t v9; // r14
  const char *v10; // r15
  __int64 v11; // rdx
  const char *v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v18; // [rsp+0h] [rbp-30h]
  __int64 v19; // [rsp+8h] [rbp-28h]

  v4 = (__int64)a2;
  v6 = *a1;
  if ( !v6 )
  {
    v18 = a4;
    v19 = a3;
    v14 = sub_22077B0(192);
    a3 = v19;
    a4 = v18;
    v6 = v14;
    if ( v14 )
    {
      *(_QWORD *)(v14 + 8) = 0;
      v15 = v14 + 32;
      *(_QWORD *)(v15 - 16) = 0;
      *(_QWORD *)(v15 - 8) = 0;
      *(_QWORD *)(v6 + 40) = v15;
      *(_QWORD *)(v6 + 32) = v15;
      *(_QWORD *)(v6 + 56) = v15;
      *(_QWORD *)(v6 + 96) = v6 + 88;
      *(_QWORD *)(v6 + 88) = v6 + 88;
      *(_QWORD *)(v6 + 48) = 0;
      *(_QWORD *)(v6 + 64) = 0;
      *(_QWORD *)(v6 + 72) = 0;
      *(_QWORD *)(v6 + 80) = 0;
      *(_QWORD *)v6 = &unk_49E5470;
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
    v16 = *a1;
    *a1 = v6;
    if ( v16 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
      v6 = *a1;
      a4 = v18;
      a3 = v19;
    }
  }
  sub_FE6A60(v6, (__int64)a2, a3, a4);
  LODWORD(v7) = dword_4F8E068;
  if ( dword_4F8E068 )
  {
    v8 = qword_4F8DF28[9];
    if ( !qword_4F8DF28[9]
      || (v6 = (__int64)a2, v12 = (const char *)qword_4F8DF28[8], v7 = sub_BD5D20((__int64)a2), v8 == v13)
      && (a2 = v12, v6 = (__int64)v7, LODWORD(v7) = memcmp(v7, v12, v8), !(_DWORD)v7) )
    {
      a2 = "BlockFrequencyDAGs";
      v6 = (__int64)a1;
      LODWORD(v7) = (unsigned int)sub_FE7C90((__int64)a1, "BlockFrequencyDAGs", (void *)0x12);
    }
  }
  if ( (_BYTE)qword_4F8DB48 )
  {
    v9 = qword_4F8DA08[9];
    if ( !qword_4F8DA08[9]
      || (v10 = (const char *)qword_4F8DA08[8], v7 = sub_BD5D20(v4), v9 == v11)
      && (a2 = v10, v6 = (__int64)v7, LODWORD(v7) = memcmp(v7, v10, v9), !(_DWORD)v7) )
    {
      sub_C5F790(v6, (__int64)a2);
      LODWORD(v7) = sub_FDC540(a1);
    }
  }
  return (int)v7;
}
