// Function: sub_22C4F30
// Address: 0x22c4f30
//
__int64 __fastcall sub_22C4F30(__int64 a1, unsigned int a2, __int64 a3, unsigned __int64 a4)
{
  unsigned int v5; // ebx
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int64 *v16; // r14
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // [rsp+10h] [rbp-170h] BYREF
  __int64 v20; // [rsp+18h] [rbp-168h]
  __int64 v21; // [rsp+20h] [rbp-160h]
  __int64 v22; // [rsp+30h] [rbp-150h] BYREF
  __int64 v23; // [rsp+38h] [rbp-148h]
  __int64 v24; // [rsp+40h] [rbp-140h]
  _BYTE v25[304]; // [rsp+50h] [rbp-130h] BYREF

  v5 = a2;
  if ( a2 <= 4
    || (a4 = ((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
              | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
              | (a2 - 1)
              | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
            | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 16,
        v5 = (a4
            | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
              | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
              | (a2 - 1)
              | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
            | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((a2 - 1) >> 1))
           + 1,
        v5 > 0x40) )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      if ( v5 <= 4 )
      {
        *(_BYTE *)(a1 + 8) |= 1u;
        goto LABEL_9;
      }
      v8 = (unsigned __int64)v5 << 6;
LABEL_5:
      v9 = sub_C7D670(v8, 8);
      *(_DWORD *)(a1 + 24) = v5;
      *(_QWORD *)(a1 + 16) = v9;
LABEL_9:
      v10 = v7 << 6;
      sub_22C4B10(a1, v6, v6 + v10, a4);
      return sub_C7D6A0(v6, v10, 8);
    }
    v19 = 0;
    v12 = a1 + 16;
    v13 = a1 + 272;
    v20 = 0;
    v21 = -4096;
    v22 = 0;
    v23 = 0;
    v24 = -8192;
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      v5 = 64;
      v8 = 4096;
      goto LABEL_5;
    }
    v19 = 0;
    v12 = a1 + 16;
    v13 = a1 + 272;
    v5 = 64;
    v20 = 0;
    v21 = -4096;
    v22 = 0;
    v23 = 0;
    v24 = -8192;
  }
  v14 = *(_QWORD *)(v12 + 16);
  v15 = -4096;
  v16 = (unsigned __int64 *)v25;
  if ( v14 == -4096 )
    goto LABEL_19;
LABEL_13:
  v15 = v14;
  if ( v24 != v14 )
  {
    if ( v16 )
    {
      v16[2] = v14;
      *v16 = 0;
      v16[1] = 0;
      if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
        sub_BD6050(v16, *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL);
    }
    v17 = (__int64)(v16 + 3);
    v16 += 8;
    sub_22C0650(v17, (unsigned __int8 *)(v12 + 24));
    sub_22C0090((unsigned __int8 *)(v12 + 24));
    v15 = *(_QWORD *)(v12 + 16);
  }
  while ( 1 )
  {
LABEL_19:
    LOBYTE(v14) = v15 != 0;
    if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
      sub_BD60C0((_QWORD *)v12);
    v12 += 64;
    if ( v12 == v13 )
      break;
    v15 = v21;
    v14 = *(_QWORD *)(v12 + 16);
    if ( v14 != v21 )
      goto LABEL_13;
  }
  if ( v5 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v18 = sub_C7D670((unsigned __int64)v5 << 6, 8);
    *(_DWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 16) = v18;
  }
  sub_22C4B10(a1, (__int64)v25, (__int64)v16, v14);
  sub_D68D70(&v22);
  return sub_D68D70(&v19);
}
