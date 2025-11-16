// Function: sub_603190
// Address: 0x603190
//
_QWORD *sub_603190()
{
  unsigned int v0; // r15d
  _QWORD *v1; // r12
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r14
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // [rsp+8h] [rbp-78h]
  _QWORD *v16; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-68h]
  char v18; // [rsp+1Ch] [rbp-64h]
  __int64 v19; // [rsp+20h] [rbp-60h]
  __int64 v20; // [rsp+28h] [rbp-58h]
  __int64 v21; // [rsp+30h] [rbp-50h]
  __int64 v22; // [rsp+38h] [rbp-48h]
  __int64 v23; // [rsp+40h] [rbp-40h]

  v0 = dword_4F04C3C;
  dword_4F04C3C = 1;
  if ( unk_4D0455C && unk_4D04600 <= 0x30DA3u )
    v1 = (_QWORD *)sub_736B10(10, "__pgi_tag");
  else
    v1 = (_QWORD *)sub_736B10(10, "__va_list_tag");
  sub_736B60(v1, 0, &unk_4F077C8);
  *(_BYTE *)(v1[21] + 110LL) |= 0x40u;
  v2 = *v1;
  *(_BYTE *)(*(_QWORD *)(*v1 + 96LL) + 177LL) |= 0x40u;
  v3 = v1[21];
  v16 = v1;
  v18 = 0;
  v19 = 0;
  v17 = v17 & 0xF8000000 | 1;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  *(_QWORD *)(v3 + 152) = sub_8600D0(6, 0xFFFFFFFFLL, v1, 0);
  *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 600) = &v16;
  if ( unk_4D045F0 )
  {
    v15 = sub_72BA30(5);
    v13 = sub_72CBE0();
    v14 = sub_72D2E0(v13, 0);
    sub_5F7E50("ptr_storage", v14);
    sub_5F7E50("dummy1", v14);
    sub_5F7E50("dummy2", v14);
    sub_5F7E50("dummy3", v15);
    sub_5F7E50("dummy4", v15);
  }
  else if ( unk_4D045F4 && (!unk_4D0455C || unk_4D04600 > 0x30DA3u) )
  {
    v11 = sub_72CBE0();
    v12 = sub_72D2E0(v11, 0);
    sub_5F7E50("ptr_storage", v12);
  }
  else
  {
    v4 = sub_72CBE0();
    v5 = sub_72D2E0(v4, 0);
    sub_5F7E50("ptr_storage", v5);
    sub_5F7E50("dummy1", v5);
    v6 = sub_72BA30(6);
    sub_5F7E50("dummy2", v6);
    sub_5F7E50("dummy3", v6);
  }
  sub_601910(v1, 0, (__int64)&v16, v7, v8, v9);
  *(_BYTE *)(v2 + 81) |= 2u;
  sub_863FC0();
  dword_4F04C3C = v0;
  return v1;
}
