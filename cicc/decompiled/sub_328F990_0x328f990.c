// Function: sub_328F990
// Address: 0x328f990
//
char __fastcall sub_328F990(unsigned __int64 **a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  unsigned __int64 *v7; // r12
  __int64 v8; // rdi
  unsigned int v9; // r13d
  unsigned __int64 v10; // rsi
  unsigned __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned int v14; // ebx

  v6 = *a2;
  v7 = *a1;
  if ( !*a2 )
  {
LABEL_4:
    v10 = *v7;
    if ( (*v7 & 1) != 0 )
      v10 >>= 58;
    else
      LODWORD(v10) = *(_DWORD *)(v10 + 64);
    sub_228BF90(v7, v10 + 1, 1u, a4, a5, a6);
    return 1;
  }
  v8 = *(_QWORD *)(v6 + 96);
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 <= 0x40 )
  {
    if ( !*(_QWORD *)(v8 + 24) )
      goto LABEL_4;
  }
  else if ( v9 == (unsigned int)sub_C444A0(v8 + 24) )
  {
    goto LABEL_4;
  }
  v12 = *v7;
  if ( (*v7 & 1) != 0 )
    v12 >>= 58;
  else
    LODWORD(v12) = *(_DWORD *)(v12 + 64);
  sub_228BF90(v7, v12 + 1, 0, a4, a5, a6);
  v13 = *(_QWORD *)(v6 + 96);
  v14 = *(_DWORD *)(v13 + 32);
  if ( v14 <= 0x40 )
    return *(_QWORD *)(v13 + 24) == 1;
  else
    return v14 - 1 == (unsigned int)sub_C444A0(v13 + 24);
}
