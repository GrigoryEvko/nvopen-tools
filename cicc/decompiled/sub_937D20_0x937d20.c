// Function: sub_937D20
// Address: 0x937d20
//
__int64 __fastcall sub_937D20(_QWORD **a1, __int64 a2, __int64 *a3, char a4)
{
  __int64 *v4; // r13
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // r8
  bool v11; // al
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  __int64 v16; // rdi
  __int64 v17; // r12
  __int64 v19; // [rsp+8h] [rbp-198h]
  char v20; // [rsp+8h] [rbp-198h]
  unsigned __int8 v21; // [rsp+8h] [rbp-198h]
  bool v22; // [rsp+8h] [rbp-198h]
  unsigned __int8 *v24; // [rsp+50h] [rbp-150h] BYREF
  __int64 v25; // [rsp+58h] [rbp-148h]
  unsigned __int64 v26; // [rsp+60h] [rbp-140h]
  _BYTE v27[24]; // [rsp+68h] [rbp-138h] BYREF
  unsigned __int8 *v28; // [rsp+80h] [rbp-120h] BYREF
  __int64 v29; // [rsp+88h] [rbp-118h]
  unsigned __int64 v30; // [rsp+90h] [rbp-110h]
  _BYTE v31[24]; // [rsp+98h] [rbp-108h] BYREF
  unsigned __int8 *v32; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v33; // [rsp+B8h] [rbp-E8h]
  unsigned __int64 v34; // [rsp+C0h] [rbp-E0h]
  _BYTE v35[24]; // [rsp+C8h] [rbp-D8h] BYREF
  _BYTE *v36; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+E8h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+F0h] [rbp-B0h] BYREF

  v36 = v38;
  v37 = 0x1000000000LL;
  v24 = v27;
  v28 = v31;
  v25 = 0;
  v26 = 16;
  v29 = 0;
  v30 = 16;
  v32 = v35;
  v33 = 0;
  v34 = 16;
  if ( a3 )
  {
    v4 = a3;
    do
    {
      if ( dword_4F0690C )
      {
        v6 = (*((_DWORD *)v4 + 8) >> 13) & 1;
      }
      else
      {
        v16 = v4[1];
        LOBYTE(v6) = 0;
        if ( (*(_BYTE *)(v16 + 140) & 0xFB) == 8 )
          v6 = ((unsigned int)sub_8D4C10(v16, dword_4F077C4 != 2) >> 2) & 1;
      }
      v7 = v25;
      if ( v25 + 1 > v26 )
      {
        v20 = v6;
        sub_C8D290(&v24, v27, v25 + 1, 1);
        v7 = v25;
        LOBYTE(v6) = v20;
      }
      v24[v7] = v6;
      v8 = (unsigned int)v37;
      ++v25;
      v9 = (unsigned int)v37 + 1LL;
      v10 = v4[1];
      if ( v9 > HIDWORD(v37) )
      {
        v19 = v4[1];
        sub_C8D5F0(&v36, v38, v9, 8);
        v8 = (unsigned int)v37;
        v10 = v19;
      }
      *(_QWORD *)&v36[8 * v8] = v10;
      v11 = 0;
      LODWORD(v37) = v37 + 1;
      if ( a4 )
        v11 = (v4[4] & 2) != 0;
      v12 = v29;
      if ( v29 + 1 > v30 )
      {
        v22 = v11;
        sub_C8D290(&v28, v31, v29 + 1, 1);
        v12 = v29;
        v11 = v22;
      }
      v28[v12] = v11;
      v13 = *((_BYTE *)v4 + 32);
      v14 = v33;
      ++v29;
      v15 = v13 & 1;
      if ( v33 + 1 > v34 )
      {
        v21 = v15;
        sub_C8D290(&v32, v35, v33 + 1, 1);
        v14 = v33;
        v15 = v21;
      }
      v32[v14] = v15;
      ++v33;
      v4 = (__int64 *)*v4;
    }
    while ( v4 );
  }
  v17 = sub_9378E0(a1, a2, (__int64)&v36, &v24, &v28, &v32);
  if ( v32 != v35 )
    _libc_free(v32, a2);
  if ( v28 != v31 )
    _libc_free(v28, a2);
  if ( v24 != v27 )
    _libc_free(v24, a2);
  if ( v36 != v38 )
    _libc_free(v36, a2);
  return v17;
}
