// Function: sub_183AD80
// Address: 0x183ad80
//
__int64 __fastcall sub_183AD80(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 **v16; // r15
  char v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // r13
  _QWORD *v20; // rdx
  __int64 *v21; // r12
  __int64 v22; // r13
  __int64 v23; // rdi
  unsigned int v24; // ecx
  unsigned __int64 v25; // rsi
  _QWORD *v26; // r14
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rax
  char v30; // al
  unsigned __int8 v32; // [rsp+7h] [rbp-519h]
  __int64 **v35; // [rsp+18h] [rbp-508h]
  _QWORD *v36; // [rsp+28h] [rbp-4F8h] BYREF
  __m128i v37; // [rsp+30h] [rbp-4F0h] BYREF
  char v38; // [rsp+40h] [rbp-4E0h]
  __int64 v39[10]; // [rsp+50h] [rbp-4D0h] BYREF
  char v40; // [rsp+A0h] [rbp-480h]
  __int64 v41; // [rsp+A8h] [rbp-478h]
  __int64 v42; // [rsp+370h] [rbp-1B0h]
  unsigned __int64 v43; // [rsp+378h] [rbp-1A8h]
  __int64 v44; // [rsp+3D8h] [rbp-148h]
  unsigned __int64 v45; // [rsp+3E0h] [rbp-140h]
  char v46; // [rsp+478h] [rbp-A8h]
  __int64 v47[12]; // [rsp+480h] [rbp-A0h] BYREF
  char v48; // [rsp+4E0h] [rbp-40h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F98A8D )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_29;
  }
  v32 = 0;
  v13 = *(_QWORD **)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
                       *(_QWORD *)(v11 + 8),
                       &unk_4F98A8D)
                   + 160);
  v48 = 0;
  v46 = 0;
  v39[0] = a1;
  while ( 1 )
  {
    v35 = *(__int64 ***)(a2 + 24);
    if ( *(__int64 ***)(a2 + 16) == v35 )
      break;
    v16 = *(__int64 ***)(a2 + 16);
    v17 = 0;
    do
    {
      v21 = *v16;
      v22 = **v16;
      if ( v22 )
      {
        v23 = **v16;
        v37.m128i_i64[0] = (__int64)sub_1832310;
        v37.m128i_i64[1] = (__int64)&v36;
        v36 = v13;
        v24 = *(_DWORD *)(a1 + 156);
        v38 = 1;
        v25 = sub_183A180(v23, sub_1833CC0, (__int64)v39, v24, &v37, a3, a4, a5, a6, v14, v15, a9, a10);
        if ( v25 )
        {
          v26 = (_QWORD *)sub_1399010(v13, v25);
          v27 = v26[1];
          v28 = v26[2];
          v29 = v26[3];
          v26[1] = v21[1];
          v26[2] = v21[2];
          v26[3] = v21[3];
          v21[3] = v29;
          LODWORD(v29) = *((_DWORD *)v21 + 8);
          v21[1] = v27;
          v21[2] = v28;
          if ( (_DWORD)v29 )
          {
            v30 = *(_BYTE *)(v22 + 32);
            *(_BYTE *)(v22 + 32) = v30 & 0xF0;
            if ( (v30 & 0x30) != 0 )
              *(_BYTE *)(v22 + 33) |= 0x40u;
          }
          else
          {
            v18 = (_QWORD *)sub_13977A0(v13, (unsigned __int64 *)v21);
            v19 = (__int64)v18;
            if ( v18 )
            {
              sub_15E3C20(v18);
              sub_1648B90(v19);
            }
          }
          v20 = v26;
          v17 = 1;
          sub_384E280(a2, v21, v20);
        }
      }
      ++v16;
    }
    while ( v35 != v16 );
    if ( !v17 )
      break;
    v32 = v17;
  }
  if ( v48 )
  {
    sub_134CA00(v47);
    if ( !v46 )
      return v32;
  }
  else if ( !v46 )
  {
    return v32;
  }
  if ( v45 != v44 )
    _libc_free(v45);
  if ( v43 != v42 )
    _libc_free(v43);
  if ( (v40 & 1) == 0 )
    j___libc_free_0(v41);
  return v32;
}
