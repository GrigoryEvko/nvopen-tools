// Function: sub_27A3590
// Address: 0x27a3590
//
__int64 __fastcall sub_27A3590(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  int v10; // ecx
  __int64 v11; // rsi
  int v12; // ecx
  unsigned int v13; // edx
  unsigned __int8 **v14; // rax
  unsigned __int8 *v15; // rdi
  __int64 v16; // r15
  unsigned int v17; // r12d
  __int64 *v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int v23; // eax
  int v24; // r10d

  v8 = *(_QWORD *)(a1 + 248);
  v10 = *(_DWORD *)(v8 + 56);
  v11 = *(_QWORD *)(v8 + 40);
  if ( !v10 )
    return sub_27A3120(a1, a2, a3, 0);
  v12 = v10 - 1;
  v13 = v12 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v14 = (unsigned __int8 **)(v11 + 16LL * v13);
  v15 = *v14;
  if ( a3 != *v14 )
  {
    v23 = 1;
    while ( v15 != (unsigned __int8 *)-4096LL )
    {
      v24 = v23 + 1;
      v13 = v12 & (v23 + v13);
      v14 = (unsigned __int8 **)(v11 + 16LL * v13);
      v15 = *v14;
      if ( a3 == *v14 )
        goto LABEL_3;
      v23 = v24;
    }
    return sub_27A3120(a1, a2, a3, 0);
  }
LABEL_3:
  v16 = (__int64)v14[1];
  if ( !v16 )
    return sub_27A3120(a1, a2, a3, 0);
  if ( (_BYTE)a5 )
    sub_D75590(*(__int64 **)(a1 + 256), (__int64 *)v14[1], a4, 2, a5, a4);
  v17 = sub_27A3120(a1, a2, a3, v16);
  sub_27A32A0(a1, v16, v18, v19, v20, v21);
  return v17;
}
