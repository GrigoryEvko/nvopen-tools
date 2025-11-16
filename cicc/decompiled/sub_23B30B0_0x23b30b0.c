// Function: sub_23B30B0
// Address: 0x23b30b0
//
void __fastcall sub_23B30B0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rsi
  unsigned __int64 v3; // r15
  __int64 v4; // r13
  char v5; // al
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdx
  char *v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // rdx
  unsigned int i; // r15d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // [rsp+28h] [rbp-108h]
  __int64 v25; // [rsp+38h] [rbp-F8h]
  __int64 v26; // [rsp+40h] [rbp-F0h]
  int v27; // [rsp+48h] [rbp-E8h]
  _BYTE *v28; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v29; // [rsp+58h] [rbp-D8h]
  __int64 v30; // [rsp+60h] [rbp-D0h]
  _BYTE v31[24]; // [rsp+68h] [rbp-C8h] BYREF
  const char *v32; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+88h] [rbp-A8h]
  __int64 v34; // [rsp+90h] [rbp-A0h]
  __int64 v35; // [rsp+98h] [rbp-98h]
  __int64 v36; // [rsp+A0h] [rbp-90h]
  __int64 v37; // [rsp+A8h] [rbp-88h]
  _BYTE **v38; // [rsp+B0h] [rbp-80h]
  char *v39; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v40; // [rsp+C8h] [rbp-68h]
  _QWORD v41[2]; // [rsp+D0h] [rbp-60h] BYREF
  char v42; // [rsp+E0h] [rbp-50h]
  _QWORD v43[2]; // [rsp+E8h] [rbp-48h] BYREF
  _QWORD *v44; // [rsp+F8h] [rbp-38h] BYREF

  v2 = (_QWORD *)(a2 + 48);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0x2800000000LL;
  v3 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v3 == v2 )
    goto LABEL_30;
  if ( !v3 )
    BUG();
  v4 = v3 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
LABEL_30:
    BUG();
  v5 = *(_BYTE *)(v3 - 24);
  if ( v5 == 31 )
  {
    v7 = *(_QWORD *)(v3 - 56);
    if ( (*(_DWORD *)(v3 - 20) & 0x7FFFFFF) == 1 )
    {
      v32 = sub_BD5D20(v7);
      v33 = v20;
      sub_95CA80((__int64 *)&v39, (__int64)&v32);
      v10 = (char *)byte_3F871B3;
      v11 = 0;
    }
    else
    {
      v32 = sub_BD5D20(v7);
      v33 = v8;
      sub_95CA80((__int64 *)&v39, (__int64)&v32);
      sub_23B2900((__int64)a1, (__int64)v39, v40, (__int64)"true", 4);
      sub_2240A30((unsigned __int64 *)&v39);
      v32 = sub_BD5D20(*(_QWORD *)(v3 - 88));
      v33 = v9;
      sub_95CA80((__int64 *)&v39, (__int64)&v32);
      v10 = "false";
      v11 = 5;
    }
    sub_23B2900((__int64)a1, (__int64)v39, v40, (__int64)v10, v11);
    sub_2240A30((unsigned __int64 *)&v39);
  }
  else if ( v5 == 32 )
  {
    v32 = sub_BD5D20(*(_QWORD *)(*(_QWORD *)(v3 - 32) + 32LL));
    v33 = v12;
    sub_95CA80((__int64 *)&v39, (__int64)&v32);
    sub_23B2900((__int64)a1, (__int64)v39, v40, (__int64)"default", 7);
    sub_2240A30((unsigned __int64 *)&v39);
    v24 = ((*(_DWORD *)(v3 - 20) & 0x7FFFFFFu) >> 1) - 1;
    if ( (*(_DWORD *)(v3 - 20) & 0x7FFFFFFu) >> 1 != 1 )
    {
      v13 = 0;
      do
      {
        v17 = *(_QWORD *)(*(_QWORD *)(v3 - 32) + 32LL * (unsigned int)(2 * ++v13));
        v18 = *(__int64 **)(v17 + 24);
        v19 = *(_DWORD *)(v17 + 32);
        if ( v19 <= 0x40 )
        {
          v14 = 0;
          if ( v19 )
            v14 = (__int64)((_QWORD)v18 << (64 - (unsigned __int8)v19)) >> (64 - (unsigned __int8)v19);
        }
        else
        {
          v14 = *v18;
        }
        v43[1] = v14;
        v39 = "{0}";
        v41[0] = &v44;
        v40 = 3;
        v43[0] = &unk_4A16058;
        v41[1] = 1;
        v44 = v43;
        v42 = 1;
        v28 = v31;
        v37 = 0x100000000LL;
        v29 = 0;
        v34 = 0;
        v32 = (const char *)&unk_49DD288;
        v30 = 20;
        v38 = &v28;
        v33 = 2;
        v35 = 0;
        v36 = 0;
        sub_CB5980((__int64)&v32, 0, 0, 0);
        sub_CB6840((__int64)&v32, (__int64)&v39);
        v32 = (const char *)&unk_49DD388;
        sub_CB5840((__int64)&v32);
        v15 = 32;
        if ( (_DWORD)v13 != -1 )
          v15 = 32LL * (unsigned int)(2 * v13 + 1);
        v25 = (__int64)v28;
        v26 = v29;
        v32 = sub_BD5D20(*(_QWORD *)(*(_QWORD *)(v3 - 32) + v15));
        v33 = v16;
        sub_95CA80((__int64 *)&v39, (__int64)&v32);
        sub_23B2900((__int64)a1, (__int64)v39, v40, v25, v26);
        if ( v39 != (char *)v41 )
          j_j___libc_free_0((unsigned __int64)v39);
        if ( v28 != v31 )
          _libc_free((unsigned __int64)v28);
      }
      while ( v24 != v13 );
    }
  }
  else
  {
    v27 = sub_B46E30(v3 - 24);
    if ( v27 )
    {
      for ( i = 0; i != v27; ++i )
      {
        v22 = sub_B46EC0(v4, i);
        v32 = sub_BD5D20(v22);
        v33 = v23;
        sub_95CA80((__int64 *)&v39, (__int64)&v32);
        sub_23B2900((__int64)a1, (__int64)v39, v40, (__int64)byte_3F871B3, 0);
        if ( v39 != (char *)v41 )
          j_j___libc_free_0((unsigned __int64)v39);
      }
    }
  }
}
