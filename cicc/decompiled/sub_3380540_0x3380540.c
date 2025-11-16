// Function: sub_3380540
// Address: 0x3380540
//
void __fastcall sub_3380540(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // r14
  _QWORD *v13; // rbx
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rsi
  int v21; // eax
  int v22; // r9d
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  unsigned int v26; // [rsp+28h] [rbp-58h]
  unsigned int v27; // [rsp+2Ch] [rbp-54h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+38h] [rbp-48h]
  __int64 v30[7]; // [rsp+48h] [rbp-38h] BYREF

  v28 = a3;
  v5 = *(unsigned int *)(a1 + 96);
  v6 = *(_QWORD *)(a1 + 80);
  v29 = a4;
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
      {
        v23 = *(_QWORD *)(a1 + 104) + 32LL * *((unsigned int *)v8 + 2);
        if ( v23 != *(_QWORD *)(a1 + 104) + 32LL * *(unsigned int *)(a1 + 112) )
        {
          v24 = *(_QWORD *)(v23 + 16);
          if ( *(_QWORD *)(v23 + 8) != v24 )
          {
            v10 = *(_QWORD *)(v23 + 8);
            do
            {
              v11 = *(_QWORD *)(v10 + 24);
              v30[0] = v11;
              if ( v11 )
                sub_B96E90((__int64)v30, v11, 1);
              v12 = *(_QWORD *)(v10 + 8);
              v13 = *(_QWORD **)(v10 + 16);
              v27 = *(_DWORD *)v10;
              v26 = *(_DWORD *)(v28 + 72);
              v14 = sub_B10CD0((__int64)v30);
              if ( !sub_337F9F0(a1, a2, v12, v13, v14, 0, &v28) )
              {
                v15 = v26;
                if ( v27 >= v26 )
                  v15 = v27;
                v16 = sub_3374700(a1, v28, v29, v12, (int)v13, (__int64)v30, v15);
                sub_33F99B0(*(_QWORD *)(a1 + 864), v16, 0);
              }
              if ( v30[0] )
                sub_B91220((__int64)v30, v30[0]);
              v10 += 32;
            }
            while ( v24 != v10 );
            v17 = *(_QWORD *)(v23 + 8);
            v18 = *(_QWORD *)(v23 + 16);
            if ( v17 != v18 )
            {
              v19 = *(_QWORD *)(v23 + 8);
              do
              {
                v20 = *(_QWORD *)(v19 + 24);
                if ( v20 )
                  sub_B91220(v19 + 24, v20);
                v19 += 32;
              }
              while ( v18 != v19 );
              *(_QWORD *)(v23 + 16) = v17;
            }
          }
        }
      }
    }
    else
    {
      v21 = 1;
      while ( v9 != -4096 )
      {
        v22 = v21 + 1;
        v7 = (v5 - 1) & (v21 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_3;
        v21 = v22;
      }
    }
  }
}
