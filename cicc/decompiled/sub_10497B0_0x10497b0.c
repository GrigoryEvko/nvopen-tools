// Function: sub_10497B0
// Address: 0x10497b0
//
__int64 *__fastcall sub_10497B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 *v8; // r13
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  int v15; // r11d
  unsigned int i; // eax
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // r14
  __int64 *v20; // rax
  __int64 v21; // [rsp+0h] [rbp-C0h]
  __int64 v22; // [rsp+8h] [rbp-B8h]
  __int64 v23; // [rsp+10h] [rbp-B0h] BYREF
  char v24; // [rsp+90h] [rbp-30h] BYREF

  v4 = 0;
  v8 = (__int64 *)sub_B2BE50(a3);
  if ( (unsigned __int8)sub_B6E900((__int64)v8) )
  {
    v4 = sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8;
    if ( (unsigned __int8)sub_B6E980((__int64)v8) )
    {
      v10 = sub_BC1CD0(a4, &unk_4F82410, a3);
      v11 = *(_QWORD *)(a3 + 40);
      v12 = *(_QWORD *)(v10 + 8);
      v13 = *(unsigned int *)(v12 + 88);
      v14 = *(_QWORD *)(v12 + 72);
      if ( (_DWORD)v13 )
      {
        v15 = 1;
        for ( i = (v13 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
                    | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; i = (v13 - 1) & v18 )
        {
          v17 = v14 + 24LL * i;
          if ( *(_UNKNOWN **)v17 == &unk_4F87C68 && v11 == *(_QWORD *)(v17 + 8) )
            break;
          if ( *(_QWORD *)v17 == -4096 && *(_QWORD *)(v17 + 8) == -4096 )
            goto LABEL_2;
          v18 = v15 + i;
          ++v15;
        }
        if ( v17 != v14 + 24 * v13 )
        {
          v19 = *(_QWORD *)(*(_QWORD *)(v17 + 16) + 24LL);
          if ( v19 )
          {
            v22 = 1;
            v20 = &v23;
            do
            {
              *v20 = -4096;
              v20 += 2;
            }
            while ( v20 != (__int64 *)&v24 );
            LOBYTE(v22) = 1;
            v21 = sub_D844E0(v19 + 8);
            sub_B6E910(v8, v21, v22);
          }
        }
      }
    }
  }
LABEL_2:
  *a1 = a3;
  a1[1] = v4;
  a1[2] = 0;
  return a1;
}
