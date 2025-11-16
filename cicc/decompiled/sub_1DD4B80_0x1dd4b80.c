// Function: sub_1DD4B80
// Address: 0x1dd4b80
//
__int64 __fastcall sub_1DD4B80(__int64 a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  __int64 v4; // r14
  __int64 v5; // rdi
  int v6; // ebx
  unsigned __int64 v7; // r15
  __int64 (*v8)(); // rax
  char v10; // al
  int v11; // r8d
  int v12; // r9d
  unsigned __int64 v13; // rax
  int v14; // ebx
  __int64 v15; // rdx
  _QWORD *v16; // rax
  _QWORD *i; // rdx

  v3 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v3 == sub_1D00B10 )
    BUG();
  v4 = *(_QWORD *)(a2 + 56);
  v5 = v3();
  v6 = *(_DWORD *)(v4 + 32);
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(v4 + 16) - *(_QWORD *)(v4 + 8)) >> 3);
  v8 = *(__int64 (**)())(*(_QWORD *)v5 + 312LL);
  if ( v8 != sub_1DD2760 )
  {
    v10 = ((__int64 (__fastcall *)(__int64, __int64))v8)(v5, a2);
    if ( v6 != (_DWORD)v7 && v10 == 1 )
    {
      v13 = *(unsigned int *)(a1 + 240);
      v14 = -858993459 * ((__int64)(*(_QWORD *)(v4 + 16) - *(_QWORD *)(v4 + 8)) >> 3) - *(_DWORD *)(v4 + 32);
      if ( v14 >= v13 )
      {
        if ( v14 <= v13 )
        {
LABEL_15:
          sub_1DD45C0(a1, a2);
          *(_BYTE *)(v4 + 652) = sub_1DD32B0(a1, (_QWORD *)a2);
          return 1;
        }
        if ( v14 > (unsigned __int64)*(unsigned int *)(a1 + 244) )
        {
          sub_16CD150(a1 + 232, (const void *)(a1 + 248), v14, 8, v11, v12);
          v13 = *(unsigned int *)(a1 + 240);
        }
        v15 = *(_QWORD *)(a1 + 232);
        v16 = (_QWORD *)(v15 + 8 * v13);
        for ( i = (_QWORD *)(v15 + 8LL * v14); i != v16; ++v16 )
        {
          if ( v16 )
            *v16 = 0;
        }
      }
      *(_DWORD *)(a1 + 240) = v14;
      goto LABEL_15;
    }
  }
  return 1;
}
