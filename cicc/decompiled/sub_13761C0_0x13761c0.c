// Function: sub_13761C0
// Address: 0x13761c0
//
__int64 __fastcall sub_13761C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rsi
  char **v7; // rbx
  char *v8; // r10
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  char *v12; // rsi
  __int64 v13; // r11
  __int64 v14; // rdx
  char *v16; // [rsp+0h] [rbp-90h]
  __int64 v17; // [rsp+8h] [rbp-88h]
  char **v19; // [rsp+18h] [rbp-78h]
  char *v20; // [rsp+20h] [rbp-70h] BYREF
  __int64 v21; // [rsp+28h] [rbp-68h]
  __int64 v22; // [rsp+30h] [rbp-60h]
  __int64 v23; // [rsp+38h] [rbp-58h]
  char *v24; // [rsp+40h] [rbp-50h] BYREF
  __int64 v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h]
  __int64 v27; // [rsp+58h] [rbp-38h]

  if ( a1 != a2 )
  {
    v4 = a1;
    do
    {
      if ( a3 )
      {
        *(_DWORD *)a3 = *(_DWORD *)v4;
        *(_DWORD *)(a3 + 4) = *(_DWORD *)(v4 + 4);
        v5 = ((__int64)(*(_QWORD *)(v4 + 56) - *(_QWORD *)(v4 + 64)) >> 3)
           + ((((__int64)(*(_QWORD *)(v4 + 80) - *(_QWORD *)(v4 + 48)) >> 3) - 1) << 6);
        v6 = *(_QWORD *)(v4 + 40) - *(_QWORD *)(v4 + 24);
        *(_QWORD *)(a3 + 8) = 0;
        *(_QWORD *)(a3 + 16) = 0;
        *(_QWORD *)(a3 + 24) = 0;
        *(_QWORD *)(a3 + 32) = 0;
        *(_QWORD *)(a3 + 40) = 0;
        *(_QWORD *)(a3 + 48) = 0;
        *(_QWORD *)(a3 + 56) = 0;
        *(_QWORD *)(a3 + 64) = 0;
        *(_QWORD *)(a3 + 72) = 0;
        *(_QWORD *)(a3 + 80) = 0;
        sub_1371810((__int64 *)(a3 + 8), v5 + (v6 >> 3));
        v7 = *(char ***)(v4 + 48);
        v8 = *(char **)(a3 + 24);
        v9 = *(_QWORD *)(a3 + 32);
        v10 = *(_QWORD *)(a3 + 40);
        v11 = *(_QWORD *)(a3 + 48);
        v17 = *(_QWORD *)(v4 + 56);
        v12 = *(char **)(v4 + 24);
        v13 = *(_QWORD *)(v4 + 40);
        v16 = *(char **)(v4 + 64);
        v19 = *(char ***)(v4 + 80);
        if ( v19 == v7 )
        {
          v27 = *(_QWORD *)(a3 + 48);
          v25 = v9;
          v26 = v10;
          v24 = v8;
          sub_1376090(&v20, v12, v17, &v24);
        }
        else
        {
          v22 = *(_QWORD *)(a3 + 40);
          v23 = v11;
          v14 = v13;
          v20 = v8;
          v21 = v9;
          while ( 1 )
          {
            ++v7;
            sub_1376090(&v24, v12, v14, &v20);
            if ( v19 == v7 )
              break;
            v22 = v26;
            v23 = v27;
            v20 = v24;
            v21 = v25;
            v12 = *v7;
            v14 = (__int64)(*v7 + 512);
          }
          sub_1376090(&v20, v16, v17, &v24);
        }
      }
      v4 += 88;
      a3 += 88;
    }
    while ( a2 != v4 );
  }
  return a3;
}
