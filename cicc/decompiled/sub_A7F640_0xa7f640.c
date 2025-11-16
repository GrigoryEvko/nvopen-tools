// Function: sub_A7F640
// Address: 0xa7f640
//
__int64 __fastcall sub_A7F640(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5, _BYTE *a6, char a7)
{
  _BYTE *v8; // r14
  char v10; // r10
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // r13d
  unsigned int v14; // ebx
  char v15; // r10
  unsigned int v16; // esi
  int v17; // edi
  unsigned int v18; // eax
  __int64 v19; // rcx
  unsigned int v20; // edx
  __int64 v21; // rdi
  unsigned int v22; // ebx
  __int64 (__fastcall *v23)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v24; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int *v28; // r14
  __int64 v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // rsi
  char *v34; // [rsp+20h] [rbp-190h] BYREF
  char v35; // [rsp+40h] [rbp-170h]
  char v36; // [rsp+41h] [rbp-16Fh]
  char v37; // [rsp+50h] [rbp-160h] BYREF
  __int16 v38; // [rsp+70h] [rbp-140h]
  _DWORD v39[76]; // [rsp+80h] [rbp-130h] BYREF

  v8 = (_BYTE *)a2;
  v10 = a7;
  v11 = *(_QWORD *)(a4 + 24);
  if ( *(_DWORD *)(a4 + 32) > 0x40u )
    v11 = **(_QWORD **)(a4 + 24);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_DWORD *)(v12 + 32);
  v14 = v11 & (v13 - 1);
  if ( !a7 )
    v14 = v11;
  if ( v14 <= 0x1F )
  {
    if ( v14 > 0x10 )
    {
      a3 = (_BYTE *)a2;
      v14 -= 16;
      v26 = sub_AD6530(v12);
      v10 = a7;
      v8 = (_BYTE *)v26;
    }
    if ( v13 )
    {
      v15 = v10 ^ 1;
      v16 = 0;
      v17 = -v14;
      do
      {
        v18 = v14;
        do
        {
          if ( v18 <= 0xF || (v20 = v13 - 16 + v18, !v15) )
            v20 = v18;
          v19 = v18 + v17;
          ++v18;
          v39[v19] = v16 + v20;
        }
        while ( v18 != v14 + 16 );
        v16 += 16;
        v17 += 16;
      }
      while ( v13 > v16 );
    }
    v21 = *(_QWORD *)(a1 + 80);
    v22 = v13;
    v36 = 1;
    v34 = "palignr";
    v35 = 3;
    v23 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v21 + 112LL);
    if ( v23 == sub_9B6630 )
    {
      if ( *a3 > 0x15u || *v8 > 0x15u )
        goto LABEL_23;
      v24 = sub_AD5CE0(a3, v8, v39, v13, 0, a6);
    }
    else
    {
      v24 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, _QWORD))v23)(v21, a3, v8, v39, v13);
    }
    if ( v24 )
      return sub_A7EE20(a1, a6, v24, a5);
LABEL_23:
    v38 = 257;
    v27 = sub_BD2C40(112, unk_3F1FE60);
    v24 = v27;
    if ( v27 )
      sub_B4E9E0(v27, (_DWORD)a3, (_DWORD)v8, (unsigned int)v39, v22, (unsigned int)&v37, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v24,
      &v34,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v28 = *(unsigned int **)a1;
    v29 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v29 )
    {
      do
      {
        v30 = *((_QWORD *)v28 + 1);
        v31 = *v28;
        v28 += 4;
        sub_B99FD0(v24, v31, v30);
      }
      while ( (unsigned int *)v29 != v28 );
    }
    return sub_A7EE20(a1, a6, v24, a5);
  }
  return sub_AD6530(v12);
}
