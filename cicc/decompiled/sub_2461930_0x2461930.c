// Function: sub_2461930
// Address: 0x2461930
//
_QWORD *__fastcall sub_2461930(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v7; // r15
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  int v14; // r9d
  unsigned int i; // eax
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // r11
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-D8h]
  __int64 v22; // [rsp+10h] [rbp-D0h]
  _QWORD *v23; // [rsp+18h] [rbp-C8h]
  __int64 v24; // [rsp+30h] [rbp-B0h] BYREF
  char v25; // [rsp+B0h] [rbp-30h] BYREF

  v7 = a1 + 4;
  v23 = a1 + 10;
  if ( sub_B2FC80(a3) )
  {
    a1[1] = v7;
    a1[6] = 0;
    a1[7] = a1 + 10;
  }
  else
  {
    v9 = sub_BC1CD0(a4, &unk_4F82410, a3);
    v10 = *(_QWORD *)(a3 + 40);
    v11 = *(_QWORD *)(v9 + 8);
    v12 = *(unsigned int *)(v11 + 88);
    v13 = *(_QWORD *)(v11 + 72);
    if ( !(_DWORD)v12 )
      goto LABEL_19;
    v14 = 1;
    for ( i = (v12 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = (v12 - 1) & v17 )
    {
      v16 = v13 + 24LL * i;
      if ( *(_UNKNOWN **)v16 == &unk_4F87C68 && v10 == *(_QWORD *)(v16 + 8) )
        break;
      if ( *(_QWORD *)v16 == -4096 && *(_QWORD *)(v16 + 8) == -4096 )
        goto LABEL_19;
      v17 = v14 + i;
      ++v14;
    }
    if ( v16 == v13 + 24 * v12 )
    {
LABEL_19:
      v18 = 0;
    }
    else
    {
      v18 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL);
      if ( v18 )
      {
        v18 += 8;
        v19 = &v24;
        do
        {
          *v19 = -4096;
          v19 += 2;
        }
        while ( v19 != (__int64 *)&v25 );
      }
    }
    v21 = v18;
    v22 = sub_BC1CD0(a4, &unk_4F8D9A8, a3);
    v20 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
    if ( (unsigned __int8)sub_24606E0(a3, (__int64 *)(v22 + 8), v21, (__int64 *)(v20 + 8), a2) )
    {
      memset(a1, 0, 0x60u);
      a1[1] = v7;
      *((_DWORD *)a1 + 4) = 2;
      *((_BYTE *)a1 + 28) = 1;
      a1[7] = v23;
      *((_DWORD *)a1 + 16) = 2;
      *((_BYTE *)a1 + 76) = 1;
      return a1;
    }
    a1[1] = v7;
    a1[6] = 0;
    a1[7] = v23;
  }
  a1[8] = 2;
  a1[2] = 0x100000002LL;
  *((_DWORD *)a1 + 18) = 0;
  *((_BYTE *)a1 + 76) = 1;
  *((_DWORD *)a1 + 6) = 0;
  *((_BYTE *)a1 + 28) = 1;
  a1[4] = &qword_4F82400;
  *a1 = 1;
  return a1;
}
