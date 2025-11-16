// Function: sub_3587B20
// Address: 0x3587b20
//
__int64 __fastcall sub_3587B20(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  unsigned __int8 v8; // al
  __int64 v9; // r15
  unsigned int v10; // ecx
  int v11; // eax
  __int64 (__fastcall **v12)(); // rax
  __int64 v14; // r15
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 **v17; // r14
  __int64 v18; // rax
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r13
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // [rsp+8h] [rbp-228h] BYREF
  unsigned int v25; // [rsp+Ch] [rbp-224h] BYREF
  __int64 v26[2]; // [rsp+10h] [rbp-220h] BYREF
  char v27; // [rsp+20h] [rbp-210h]
  __int64 v28[4]; // [rsp+30h] [rbp-200h] BYREF
  _QWORD v29[10]; // [rsp+50h] [rbp-1E0h] BYREF
  unsigned __int64 *v30; // [rsp+A0h] [rbp-190h]
  unsigned int v31; // [rsp+A8h] [rbp-188h]
  char v32; // [rsp+B0h] [rbp-180h] BYREF

  v5 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a2 + 16LL))(a2, a3);
  if ( v5 && *(_QWORD *)(a3 + 56) )
  {
    v6 = v5;
    v7 = sub_B10CD0(a3 + 56);
    v24 = sub_C1B040(v7);
    v8 = *(_BYTE *)(v7 - 16);
    if ( LOBYTE(qword_4F813A8[8]) )
    {
      if ( (v8 & 2) != 0 )
        v9 = *(_QWORD *)(v7 - 32);
      else
        v9 = v7 - 16 - 8LL * ((v8 >> 2) & 0xF);
      v10 = 0;
      if ( **(_BYTE **)v9 == 20 )
        v10 = *(_DWORD *)(*(_QWORD *)v9 + 4LL);
    }
    else
    {
      if ( (v8 & 2) != 0 )
        v14 = *(_QWORD *)(v7 - 32);
      else
        v14 = v7 - 16 - 8LL * ((v8 >> 2) & 0xF);
      v10 = 0;
      if ( **(_BYTE **)v14 == 20 )
      {
        v15 = *(_DWORD *)(*(_QWORD *)v14 + 4LL);
        if ( (v15 & 7) == 7 && (v15 & 0xFFFFFFF8) != 0 )
        {
          if ( (v15 & 0x10000000) != 0 )
            v10 = BYTE2(v15) & 7;
          else
            v10 = (unsigned __int16)(v15 >> 3);
        }
        else if ( (v15 & 1) != 0 )
        {
          v10 = 0;
        }
        else
        {
          v10 = (v15 >> 1) & 0x1F;
          if ( ((v15 >> 1) & 0x20) != 0 )
            v10 |= (v15 >> 2) & 0xFE0;
        }
      }
    }
    v25 = v10;
    sub_35845F0((__int64)v26, v6, v24, v10);
    if ( (v27 & 1) != 0 )
      goto LABEL_9;
    if ( sub_2A61A10((__int64)(a2 + 136), v6, v24, v25, v26[0]) )
    {
      v17 = (__int64 **)a2[161];
      v28[0] = a3;
      v28[2] = (__int64)&v24;
      v28[1] = (__int64)v26;
      v28[3] = (__int64)&v25;
      v18 = sub_B2BE50(**v17);
      if ( sub_B6EA50(v18)
        || (v22 = sub_B2BE50(**v17),
            v23 = sub_B6F970(v22),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v23 + 48LL))(v23)) )
      {
        sub_35876F0((__int64)v29, v28);
        sub_2EAFC50(v17, (__int64)v29);
        v19 = v30;
        v29[0] = &unk_49D9D40;
        v20 = &v30[10 * v31];
        if ( v30 != v20 )
        {
          do
          {
            v20 -= 10;
            v21 = v20[4];
            if ( (unsigned __int64 *)v21 != v20 + 6 )
              j_j___libc_free_0(v21);
            if ( (unsigned __int64 *)*v20 != v20 + 2 )
              j_j___libc_free_0(*v20);
          }
          while ( v19 != v20 );
          v20 = v30;
        }
        if ( v20 != (unsigned __int64 *)&v32 )
          _libc_free((unsigned __int64)v20);
      }
    }
    if ( (v27 & 1) != 0 )
    {
LABEL_9:
      v11 = v26[0];
      *(_BYTE *)(a1 + 16) |= 1u;
      *(_DWORD *)a1 = v11;
      *(_QWORD *)(a1 + 8) = v26[1];
    }
    else
    {
      v16 = v26[0];
      *(_BYTE *)(a1 + 16) &= ~1u;
      *(_QWORD *)a1 = v16;
    }
  }
  else
  {
    v12 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v12;
  }
  return a1;
}
