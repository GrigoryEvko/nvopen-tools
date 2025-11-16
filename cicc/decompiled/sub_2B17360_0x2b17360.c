// Function: sub_2B17360
// Address: 0x2b17360
//
void __fastcall sub_2B17360(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r10
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r11
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v23; // [rsp+8h] [rbp-88h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  char *v26[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v27[96]; // [rsp+30h] [rbp-60h] BYREF

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a5;
      if ( a4 + a5 == 2 )
      {
        v14 = a1;
        v8 = (__int64)a2;
LABEL_21:
        v17 = *(unsigned int *)(v14 + 8);
        if ( *(_DWORD *)(v8 + 8) > (unsigned int)v17 )
        {
          v26[0] = v27;
          v26[1] = (char *)0x600000000LL;
          if ( (_DWORD)v17 )
            sub_2B0F6D0((__int64)v26, (char **)v14, v17, a4, a5, v7);
          sub_2B0F6D0(v14, (char **)v8, v17, a4, a5, v7);
          sub_2B0F6D0(v8, v26, v18, v19, v20, v21);
          if ( v26[0] != v27 )
            _libc_free((unsigned __int64)v26[0]);
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_14;
LABEL_5:
        v8 = (__int64)a2;
        v9 = v5 / 2;
        v10 = (a3 - (__int64)a2) >> 6;
        v11 = v6 + ((v5 / 2) << 6);
        while ( v10 > 0 )
        {
          while ( 1 )
          {
            a4 = v10 >> 1;
            v12 = v8 + (v10 >> 1 << 6);
            if ( *(_DWORD *)(v12 + 8) <= *(_DWORD *)(v11 + 8) )
              break;
            v8 = v12 + 64;
            v10 = v10 - a4 - 1;
            if ( v10 <= 0 )
              goto LABEL_9;
          }
          v10 >>= 1;
        }
LABEL_9:
        v13 = (v8 - (__int64)a2) >> 6;
        while ( 1 )
        {
          v23 = v7;
          v24 = v6;
          v25 = v11;
          v14 = sub_2B11A30(v11, a2, (char **)v8, a4, a5, v7);
          sub_2B17360(v24, v25, v14, v9, v13);
          v7 = v23 - v13;
          v5 -= v9;
          if ( !v5 || !v7 )
            break;
          if ( v7 + v5 == 2 )
            goto LABEL_21;
          v6 = v14;
          a2 = (char **)v8;
          if ( v7 < v5 )
            goto LABEL_5;
LABEL_14:
          v11 = v6;
          v15 = ((__int64)a2 - v6) >> 6;
          v13 = v7 / 2;
          v8 = (__int64)&a2[8 * (v7 / 2)];
          while ( v15 > 0 )
          {
            while ( 1 )
            {
              a4 = v15 >> 1;
              v16 = v11 + (v15 >> 1 << 6);
              if ( *(_DWORD *)(v8 + 8) > *(_DWORD *)(v16 + 8) )
                break;
              v11 = v16 + 64;
              v15 = v15 - a4 - 1;
              if ( v15 <= 0 )
                goto LABEL_18;
            }
            v15 >>= 1;
          }
LABEL_18:
          v9 = (v11 - v6) >> 6;
        }
      }
    }
  }
}
