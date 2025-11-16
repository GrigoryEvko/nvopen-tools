// Function: sub_1F1FA40
// Address: 0x1f1fa40
//
void __fastcall sub_1F1FA40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  int v6; // r14d
  __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned int v10; // r10d
  __int64 *v11; // rcx
  __int64 *v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v16; // [rsp+18h] [rbp-88h] BYREF
  __int64 v17; // [rsp+20h] [rbp-80h]
  _QWORD v18[15]; // [rsp+28h] [rbp-78h] BYREF

  v6 = a4;
  v8 = a1;
  v9 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v9 )
  {
    v15 = a1;
    v17 = 0x400000000LL;
    v16 = v18;
    sub_1F17CD0((__int64)&v15, a2, v9, a4, a5, a6);
    v8 = v15;
    if ( *(_DWORD *)(v15 + 184) )
    {
      sub_1F1F320((__int64)&v15, a2, a3, v6);
      goto LABEL_15;
    }
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 188);
    if ( v10 != 9 )
    {
      if ( v10 )
      {
        v11 = (__int64 *)(a1 + 8);
        do
        {
          if ( (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(a2 >> 1) & 3) )
            break;
          LODWORD(v9) = v9 + 1;
          v11 += 2;
        }
        while ( v10 != (_DWORD)v9 );
      }
      LODWORD(v15) = v9;
      *(_DWORD *)(a1 + 188) = sub_1F15E30(a1, (unsigned int *)&v15, v10, a2, a3, v6);
      return;
    }
    v15 = a1;
    v12 = (__int64 *)(a1 + 8);
    v16 = v18;
    HIDWORD(v17) = 4;
    do
    {
      if ( (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(a2 >> 1)
                                                                                              & 3) )
        break;
      v9 = (unsigned int)(v9 + 1);
      v12 += 2;
    }
    while ( (_DWORD)v9 != 9 );
    v18[0] = a1;
    LODWORD(v17) = 1;
    v18[1] = (v9 << 32) | 9;
  }
  v13 = sub_1F15E30(v8, (unsigned int *)&v16[2 * (unsigned int)v17 - 1] + 1, *(_DWORD *)(v8 + 188), a2, a3, v6);
  if ( v13 > 9 )
  {
    v14 = sub_1F17930((_QWORD *)v8, HIDWORD(v16[2 * (unsigned int)v17 - 1]));
    sub_3945C20(&v16, v8 + 8, *(unsigned int *)(v8 + 188), v14);
    sub_1F1F320((__int64)&v15, a2, a3, v6);
  }
  else
  {
    *(_DWORD *)(v8 + 188) = v13;
    *((_DWORD *)v16 + 2) = v13;
  }
LABEL_15:
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
}
