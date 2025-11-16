// Function: sub_339CB00
// Address: 0x339cb00
//
void __fastcall sub_339CB00(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 *v5; // rdi
  int v6; // eax
  __int64 v7; // rbx
  unsigned int v8; // edx
  unsigned int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // edx
  unsigned int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int128 v19; // rax
  __int64 v20; // r12
  int v21; // edx
  int v22; // ebx
  __int64 v23; // r13
  __int64 v24; // [rsp-10h] [rbp-120h]
  __int128 v25; // [rsp-10h] [rbp-120h]
  __int64 v26; // [rsp-8h] [rbp-118h]
  __int128 v27; // [rsp+0h] [rbp-110h]
  unsigned int v28; // [rsp+20h] [rbp-F0h]
  __int64 v29; // [rsp+50h] [rbp-C0h] BYREF
  int v30; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v31[2]; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE v32[32]; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 v33[2]; // [rsp+90h] [rbp-80h] BYREF
  _BYTE v34[112]; // [rsp+A0h] [rbp-70h] BYREF

  v33[0] = (unsigned __int64)v34;
  v3 = *(_QWORD *)(a2 - 64);
  v33[1] = 0x400000000LL;
  v31[1] = 0x400000000LL;
  v4 = *(_QWORD *)(v3 + 8);
  v5 = *(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL);
  v31[0] = (unsigned __int64)v32;
  v6 = sub_2E79000(v5);
  sub_34B8FD0(*(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL), v6, v4, (unsigned int)v33, 0, (unsigned int)v31, 0);
  v7 = sub_338B750(a1, v3);
  v28 = v8;
  v9 = sub_35D5BC0(*(_QWORD *)(a1 + 968), a2, *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL), *(_QWORD *)(a2 - 32));
  v12 = *(_DWORD *)(a1 + 848);
  v29 = 0;
  v13 = v9;
  v14 = *(_QWORD *)a1;
  v15 = *(_QWORD *)(a1 + 864);
  v30 = v12;
  v16 = v24;
  v17 = v26;
  if ( v14 )
  {
    v16 = v14 + 48;
    if ( &v29 != (__int64 *)(v14 + 48) )
    {
      a2 = *(_QWORD *)(v14 + 48);
      v29 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v29, a2, 1);
    }
  }
  *(_QWORD *)&v27 = sub_33738B0(a1, a2, v16, v17, v10, v11);
  *((_QWORD *)&v27 + 1) = v18;
  *(_QWORD *)&v19 = sub_33F0B60(
                      v15,
                      v13,
                      *(unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16LL * v28),
                      *(_QWORD *)(*(_QWORD *)(v7 + 48) + 16LL * v28 + 8));
  *((_QWORD *)&v25 + 1) = v28;
  *(_QWORD *)&v25 = v7;
  v20 = sub_340F900(v15, 49, (unsigned int)&v29, 1, 0, v28, v27, v19, v25);
  v22 = v21;
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  v23 = *(_QWORD *)(a1 + 864);
  if ( v20 )
  {
    nullsub_1875(v20, *(_QWORD *)(a1 + 864), 0);
    *(_QWORD *)(v23 + 384) = v20;
    *(_DWORD *)(v23 + 392) = v22;
    sub_33E2B60(v23, 0);
  }
  else
  {
    *(_QWORD *)(v23 + 384) = 0;
    *(_DWORD *)(v23 + 392) = v22;
  }
  if ( (_BYTE *)v31[0] != v32 )
    _libc_free(v31[0]);
  if ( (_BYTE *)v33[0] != v34 )
    _libc_free(v33[0]);
}
