// Function: sub_28C1F40
// Address: 0x28c1f40
//
unsigned __int8 *__fastcall sub_28C1F40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v7; // edx
  __int64 *v8; // r14
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rax
  unsigned __int8 *v17; // r15
  __int64 v18; // r8
  __int64 v19; // rdi
  __int64 v20; // rsi
  char v21; // al
  _QWORD *v22; // rdi
  unsigned __int8 **v23; // r12
  unsigned __int8 **v24; // rbx
  unsigned __int8 *v25; // r13
  int v26; // r8d
  unsigned __int64 v29; // [rsp+10h] [rbp-70h] BYREF
  __int64 v30; // [rsp+18h] [rbp-68h]
  _QWORD v31[12]; // [rsp+20h] [rbp-60h] BYREF

  v3 = *(unsigned int *)(a1 + 72);
  v4 = *(_QWORD *)(a1 + 56);
  if ( (_DWORD)v3 )
  {
    v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v4 + 72LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v4 + 72 * v3) )
      {
        while ( 1 )
        {
          v10 = *((unsigned int *)v8 + 4);
          if ( !(_DWORD)v10 )
            break;
          v11 = 3 * v10;
          v12 = v8[1];
          v29 = 6;
          v30 = 0;
          v13 = (_QWORD *)(v12 + 8 * v11 - 24);
          v31[0] = v13[2];
          if ( v31[0] != -4096 && v31[0] != 0 && v31[0] != -8192 )
            sub_BD6050(&v29, *v13 & 0xFFFFFFFFFFFFFFF8LL);
          v14 = (unsigned int)(*((_DWORD *)v8 + 4) - 1);
          *((_DWORD *)v8 + 4) = v14;
          v15 = (_QWORD *)(v8[1] + 24 * v14);
          v16 = v15[2];
          if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
            sub_BD60C0(v15);
          v17 = (unsigned __int8 *)v31[0];
          if ( v31[0] )
          {
            if ( v31[0] != -8192 && v31[0] != -4096 )
              sub_BD60C0(&v29);
            if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 16), (__int64)v17, a3) )
            {
              v19 = *(_QWORD *)(a1 + 24);
              v20 = a2;
              v30 = 0x600000000LL;
              v29 = (unsigned __int64)v31;
              v21 = sub_D9BB00(v19, a2, v17, (__int64)&v29, v18);
              v22 = (_QWORD *)v29;
              if ( v21 )
              {
                v23 = (unsigned __int8 **)(v29 + 8LL * (unsigned int)v30);
                if ( v23 != (unsigned __int8 **)v29 )
                {
                  v24 = (unsigned __int8 **)v29;
                  do
                  {
                    v25 = *v24++;
                    sub_B44F30(v25);
                    sub_B44B50((__int64 *)v25, v20);
                    sub_B44A60((__int64)v25);
                  }
                  while ( v23 != v24 );
                  v22 = (_QWORD *)v29;
                }
                if ( v22 != v31 )
                  _libc_free((unsigned __int64)v22);
                return v17;
              }
              if ( (_QWORD *)v29 != v31 )
                _libc_free(v29);
            }
          }
        }
      }
    }
    else
    {
      v26 = 1;
      while ( v9 != -4096 )
      {
        v7 = (v3 - 1) & (v26 + v7);
        v8 = (__int64 *)(v4 + 72LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        ++v26;
      }
    }
  }
  return 0;
}
