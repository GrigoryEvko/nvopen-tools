// Function: sub_2FBD6E0
// Address: 0x2fbd6e0
//
void __fastcall sub_2FBD6E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r14d
  __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned int v10; // r10d
  __int64 *v11; // rcx
  __int64 *v12; // rcx
  unsigned int v13; // eax
  unsigned __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v18; // [rsp+18h] [rbp-88h] BYREF
  __int64 v19; // [rsp+20h] [rbp-80h]
  _QWORD v20[15]; // [rsp+28h] [rbp-78h] BYREF

  v6 = a4;
  v8 = a1;
  v9 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v9 )
  {
    v17 = a1;
    v19 = 0x400000000LL;
    v18 = v20;
    sub_2FB5930((__int64)&v17, a2, v9, a4, a5, a6);
    v8 = v17;
    if ( *(_DWORD *)(v17 + 184) )
    {
      sub_2FBCF60((unsigned int *)&v17, a2, a3, v6);
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
      LODWORD(v17) = v9;
      *(_DWORD *)(a1 + 188) = sub_2FB3800(a1, (unsigned int *)&v17, v10, a2, a3, v6);
      return;
    }
    v17 = a1;
    v12 = (__int64 *)(a1 + 8);
    v18 = v20;
    HIDWORD(v19) = 4;
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
    v20[0] = a1;
    LODWORD(v19) = 1;
    v20[1] = (v9 << 32) | 9;
  }
  v13 = sub_2FB3800(v8, (unsigned int *)&v18[2 * (unsigned int)v19 - 1] + 1, *(_DWORD *)(v8 + 188), a2, a3, v6);
  if ( v13 > 9 )
  {
    v14 = sub_2FB54B0((_QWORD *)v8, HIDWORD(v18[2 * (unsigned int)v19 - 1]));
    sub_F038C0((unsigned int *)&v18, v8 + 8, *(_DWORD *)(v8 + 188), v14, v15, v16);
    sub_2FBCF60((unsigned int *)&v17, a2, a3, v6);
  }
  else
  {
    *(_DWORD *)(v8 + 188) = v13;
    *((_DWORD *)v18 + 2) = v13;
  }
LABEL_15:
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
}
