// Function: sub_23DE7E0
// Address: 0x23de7e0
//
_QWORD *__fastcall sub_23DE7E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // r14
  unsigned int v14; // ebx
  unsigned int v15; // ebx
  __int64 v16; // r15
  __int64 v17; // rdi
  _QWORD *v18; // r14
  _BYTE *v20; // rax
  unsigned int *v21; // r14
  __int64 v22; // r15
  __int64 v23; // rdx
  unsigned int v24; // esi
  _QWORD **v25; // rdx
  int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // rsi
  unsigned int *v29; // rbx
  __int64 v30; // r12
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rdx
  int v34; // r14d
  unsigned int *v35; // rbx
  __int64 v36; // r14
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v41; // [rsp+18h] [rbp-98h]
  _BYTE v42[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v43; // [rsp+40h] [rbp-70h]
  _BYTE v44[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v45; // [rsp+70h] [rbp-40h]

  v8 = *(_DWORD *)(a1 + 120);
  v9 = *(_QWORD *)(a1 + 96);
  v43 = 257;
  v10 = sub_AD64C0(v9, (1LL << v8) - 1, 0);
  v11 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 16LL))(
          *(_QWORD *)(a2 + 80),
          28,
          a3,
          v10);
  if ( !v11 )
  {
    v45 = 257;
    v11 = sub_B504D0(28, a3, v10, (__int64)v44, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v11,
      v42,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v21 = *(unsigned int **)a2;
    v22 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v22 )
    {
      do
      {
        v23 = *((_QWORD *)v21 + 1);
        v24 = *v21;
        v21 += 4;
        sub_B99FD0(v11, v24, v23);
      }
      while ( (unsigned int *)v22 != v21 );
    }
  }
  if ( a5 > 0xF )
  {
    v45 = 257;
    v20 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a1 + 96), (a5 >> 3) - 1, 0);
    v11 = sub_929C50((unsigned int **)a2, (_BYTE *)v11, v20, (__int64)v44, 0, 0);
  }
  v43 = 257;
  v12 = *(_QWORD *)(v11 + 8);
  v13 = *(_QWORD *)(a4 + 8);
  v14 = sub_BCB060(v12);
  v15 = (v14 <= (unsigned int)sub_BCB060(v13)) + 38;
  if ( v13 == v12 )
  {
    v16 = v11;
  }
  else
  {
    v16 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
            *(_QWORD *)(a2 + 80),
            v15,
            v11,
            v13);
    if ( !v16 )
    {
      v45 = 257;
      v16 = sub_B51D30(v15, v11, v13, (__int64)v44, 0, 0);
      if ( (unsigned __int8)sub_920620(v16) )
      {
        v33 = *(_QWORD *)(a2 + 96);
        v34 = *(_DWORD *)(a2 + 104);
        if ( v33 )
          sub_B99FD0(v16, 3u, v33);
        sub_B45150(v16, v34);
      }
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v16,
        v42,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v35 = *(unsigned int **)a2;
      v36 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v36 )
      {
        do
        {
          v37 = *((_QWORD *)v35 + 1);
          v38 = *v35;
          v35 += 4;
          sub_B99FD0(v16, v38, v37);
        }
        while ( (unsigned int *)v36 != v35 );
      }
    }
  }
  v17 = *(_QWORD *)(a2 + 80);
  v43 = 257;
  v18 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v17 + 56LL))(
                    v17,
                    39,
                    v16,
                    a4);
  if ( !v18 )
  {
    v45 = 257;
    v18 = sub_BD2C40(72, unk_3F10FD0);
    if ( v18 )
    {
      v25 = *(_QWORD ***)(v16 + 8);
      v26 = *((unsigned __int8 *)v25 + 8);
      if ( (unsigned int)(v26 - 17) > 1 )
      {
        v28 = sub_BCB2A0(*v25);
      }
      else
      {
        BYTE4(v41) = (_BYTE)v26 == 18;
        LODWORD(v41) = *((_DWORD *)v25 + 8);
        v27 = (__int64 *)sub_BCB2A0(*v25);
        v28 = sub_BCE1B0(v27, v41);
      }
      sub_B523C0((__int64)v18, v28, 53, 39, v16, a4, (__int64)v44, 0, 0, 0);
    }
    (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v18,
      v42,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v29 = *(unsigned int **)a2;
    v30 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    while ( (unsigned int *)v30 != v29 )
    {
      v31 = *((_QWORD *)v29 + 1);
      v32 = *v29;
      v29 += 4;
      sub_B99FD0((__int64)v18, v32, v31);
    }
  }
  return v18;
}
