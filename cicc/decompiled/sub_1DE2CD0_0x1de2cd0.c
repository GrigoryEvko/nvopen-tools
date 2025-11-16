// Function: sub_1DE2CD0
// Address: 0x1de2cd0
//
int __fastcall sub_1DE2CD0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  __int64 v6; // rdi
  const void *v7; // rax
  size_t v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 (__fastcall *v11)(__int64, __int64); // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  _QWORD v23[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v25; // [rsp+30h] [rbp-30h]

  v4 = a2;
  v6 = *(_QWORD *)(a1 + 232);
  if ( !v6 )
  {
    v21 = a4;
    v22 = a3;
    v17 = sub_22077B0(192);
    a3 = v22;
    a4 = v21;
    v6 = v17;
    if ( v17 )
    {
      *(_QWORD *)(v17 + 8) = 0;
      v18 = v17 + 40;
      *(_QWORD *)(v18 - 24) = 0;
      *(_QWORD *)(v18 - 16) = 0;
      *(_QWORD *)(v6 + 48) = v18;
      *(_QWORD *)(v6 + 40) = v18;
      *(_QWORD *)(v6 + 32) = v18;
      *(_QWORD *)(v6 + 96) = v6 + 88;
      *(_QWORD *)(v6 + 88) = v6 + 88;
      *(_QWORD *)(v6 + 56) = 0;
      *(_QWORD *)(v6 + 64) = 0;
      *(_QWORD *)(v6 + 72) = 0;
      *(_QWORD *)(v6 + 80) = 0;
      *(_QWORD *)v6 = &unk_49FB1C8;
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
    v19 = *(_QWORD *)(a1 + 232);
    *(_QWORD *)(a1 + 232) = v6;
    if ( v19 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
      v6 = *(_QWORD *)(a1 + 232);
      a4 = v21;
      a3 = v22;
    }
  }
  LODWORD(v7) = sub_1DE24A0(v6, (__int64)a2, a3, a4);
  v8 = (unsigned int)dword_4FC4940;
  if ( dword_4FC4940 )
  {
    if ( !qword_4F983A0[21]
      || (v6 = (__int64)a2, v7 = (const void *)sub_1E0A440(a2), v16 = v15, v8 = qword_4F983A0[21], v8 == v16)
      && (!v8 || (a2 = (_QWORD *)qword_4F983A0[20], v6 = (__int64)v7, LODWORD(v7) = memcmp(v7, a2, v8), !(_DWORD)v7)) )
    {
      a2 = v24;
      v6 = a1;
      v23[0] = sub_1E0A440(v4);
      v25 = 1283;
      v23[1] = v9;
      v24[0] = "MachineBlockFrequencyDAGS.";
      v24[1] = v23;
      LODWORD(v7) = (unsigned int)sub_1DE2C60(a1, (__int64)v24, 1);
    }
  }
  if ( byte_4FC4600 )
  {
    if ( !qword_4F97E60[21]
      || (v6 = (__int64)v4, v7 = (const void *)sub_1E0A440(v4), v14 = v13, v8 = qword_4F97E60[21], v8 == v14)
      && (!v8 || (a2 = (_QWORD *)qword_4F97E60[20], v6 = (__int64)v7, LODWORD(v7) = memcmp(v7, a2, v8), !(_DWORD)v7)) )
    {
      v10 = *(_QWORD *)(a1 + 232);
      v11 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 24LL);
      v12 = sub_16BA580(v6, (__int64)a2, v8);
      LODWORD(v7) = v11(v10, v12);
    }
  }
  return (int)v7;
}
