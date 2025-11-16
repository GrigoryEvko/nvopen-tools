// Function: sub_2F74F40
// Address: 0x2f74f40
//
__int64 __fastcall sub_2F74F40(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r12d
  char v9; // bl
  __int64 (*v10)(void); // rax
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rdx
  _DWORD *v16; // r10
  __int64 v17; // rcx
  __int64 result; // rax
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int *v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  __int128 v27; // [rsp+18h] [rbp-48h]
  _DWORD *v29; // [rsp+28h] [rbp-38h]

  v8 = a2;
  v9 = *(_BYTE *)(a1 + 58);
  *((_QWORD *)&v27 + 1) = a3;
  *(_QWORD *)&v27 = a4;
  if ( v9 )
  {
    v9 = 0;
    v10 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 8) + 432LL);
    if ( v10 != sub_2F73F20 )
      v9 = v10();
  }
  v11 = *(_QWORD **)(a1 + 24);
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v11 + 16LL) + 200LL))(*(_QWORD *)(*v11 + 16LL));
  v13 = v12;
  if ( (a2 & 0x80000000) != 0LL )
  {
    v24 = v12;
    a2 = *(_QWORD *)(v11[7] + 16 * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 416LL))(v12);
    v23 = (unsigned int *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v24 + 376LL))(v24, a2);
    v16 = (_DWORD *)v26;
    v17 = *v23;
  }
  else
  {
    a2 = (unsigned int)a2;
    v25 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v12 + 424LL))(v12, (unsigned int)a2);
    v14 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v13 + 384LL))(v13, (unsigned int)a2);
    v16 = (_DWORD *)v25;
    v17 = v14;
  }
  if ( *v16 == -1 )
    v16 = 0;
  if ( v9 )
  {
    v19 = *((_QWORD *)&v27 + 1) & ~a5;
    v20 = v27 & ~a6;
    result = v20 | v19;
    if ( !(v20 | v19) )
      return result;
    v29 = v16;
    v21 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64))(**(_QWORD **)(*v11 + 16LL) + 200LL))(
            *(_QWORD *)(*v11 + 16LL),
            a2,
            v15,
            v17);
    result = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD, __int64, __int64))(*(_QWORD *)v21 + 440LL))(
               v21,
               v11,
               v8,
               v19,
               v20);
    v16 = v29;
    LODWORD(v17) = result;
  }
  else
  {
    result = a5 | a6;
    if ( a5 | a6 )
      return result;
    result = *((_QWORD *)&v27 + 1) | v27;
    if ( v27 == 0 )
      return result;
  }
  do
  {
    if ( !v16 )
      break;
    v22 = (unsigned int)*v16;
    result = *(_QWORD *)(a1 + 72);
    ++v16;
    *(_DWORD *)(result + 4 * v22) -= v17;
  }
  while ( *v16 != -1 );
  return result;
}
