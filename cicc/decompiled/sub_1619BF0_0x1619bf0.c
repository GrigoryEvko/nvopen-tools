// Function: sub_1619BF0
// Address: 0x1619bf0
//
__int64 __fastcall sub_1619BF0(__int64 a1, __int64 a2)
{
  int v3; // r12d
  unsigned int v4; // ebx
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 v9; // r14
  const char *v10; // rax
  size_t v11; // rdx
  __int64 *v12; // r12
  const char *v13; // rax
  size_t v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  char v20; // bl
  __int64 v21; // r8
  __int64 v22; // rdi
  const char *v23; // rax
  size_t v24; // rdx
  int v25; // ebx
  unsigned int v26; // r12d
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v31; // [rsp+10h] [rbp-B0h]
  __int64 v32; // [rsp+18h] [rbp-A8h]
  unsigned int v33; // [rsp+24h] [rbp-9Ch]
  __int64 *v34; // [rsp+28h] [rbp-98h]
  char v35; // [rsp+4Ah] [rbp-76h]
  unsigned __int8 v36; // [rsp+4Bh] [rbp-75h]
  unsigned int i; // [rsp+4Ch] [rbp-74h]
  __int64 v38; // [rsp+58h] [rbp-68h] BYREF
  _QWORD v39[12]; // [rsp+60h] [rbp-60h] BYREF

  if ( *(_DWORD *)(a1 + 32) )
  {
    v3 = 0;
    v4 = 0;
    do
    {
      v5 = v4++;
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v5);
      v3 |= (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 144LL))(v6, a2);
    }
    while ( *(_DWORD *)(a1 + 32) > v4 );
    v36 = v3;
  }
  else
  {
    v36 = 0;
  }
  v34 = *(__int64 **)(a2 + 40);
  v7 = sub_16033E0(*v34);
  v35 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v7 + 24LL))(v7, "size-info", 9);
  v31 = a2 + 72;
  v32 = *(_QWORD *)(a2 + 80);
  v8 = *(_DWORD *)(a1 + 32);
  if ( v32 != a2 + 72 )
  {
    do
    {
      v9 = v32 - 24;
      if ( !v32 )
        v9 = 0;
      if ( v8 )
      {
        for ( i = 0; i < v8; ++i )
        {
          v12 = *(__int64 **)(*(_QWORD *)(a1 + 24) + 8LL * i);
          v13 = (const char *)sub_1649960(v9);
          sub_160F160(a1, (__int64)v12, 0, 3, v13, v14);
          sub_1615D60(a1, v12);
          sub_1614C80(a1, (__int64)v12);
          sub_16C6860(v39);
          v39[2] = v12;
          v39[3] = v9;
          v39[0] = &unk_49ED7C0;
          v39[4] = 0;
          v15 = sub_1612E30(v12);
          v16 = v15;
          if ( v15 )
            sub_16D7910(v15);
          sub_1403F30(&v38, v12, *(_QWORD *)(a1 + 8));
          if ( v35 )
          {
            v33 = sub_160E760(a1, (__int64)v34);
            v17 = (__int64)v12;
            v20 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*v12 + 152))(v12, v9);
            sub_160FF80(a1, (__int64)v12, (__int64)v34, v33);
          }
          else
          {
            v17 = v9;
            v20 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*v12 + 152))(v12, v9);
          }
          v22 = v38;
          if ( v38 )
          {
            if ( v20 )
            {
              v17 = 2;
              (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v38 + 56LL))(v38, 2);
              v22 = v38;
            }
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v22 + 48LL))(v22);
          }
          if ( v16 )
            sub_16D7950(v16, v17, v18);
          v39[0] = &unk_49ED7C0;
          nullsub_616(v39, v17, v18, v19, v21);
          v36 |= v20;
          if ( v20 )
          {
            v23 = (const char *)sub_1649960(v9);
            sub_160F160(a1, (__int64)v12, 1, 3, v23, v24);
          }
          sub_1615E90(a1, v12);
          sub_1615FB0(a1, v12);
          nullsub_568();
          sub_16145F0(a1, (__int64)v12);
          sub_16176C0(a1, (__int64)v12);
          v10 = (const char *)sub_1649960(v9);
          sub_1615450(a1, (__int64)v12, v10, v11, 3);
          v8 = *(_DWORD *)(a1 + 32);
        }
      }
      v32 = *(_QWORD *)(v32 + 8);
    }
    while ( v31 != v32 );
  }
  v25 = 0;
  v26 = 0;
  if ( v8 )
  {
    do
    {
      v27 = v26++;
      v28 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v27);
      v25 |= (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v28 + 160LL))(v28, a2);
    }
    while ( *(_DWORD *)(a1 + 32) > v26 );
    v36 |= v25;
  }
  return v36;
}
