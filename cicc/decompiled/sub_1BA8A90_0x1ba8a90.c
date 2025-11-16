// Function: sub_1BA8A90
// Address: 0x1ba8a90
//
void __fastcall sub_1BA8A90(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 v4; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 *v8; // rax
  __int64 *v9; // r13
  __int64 *v10; // rbx
  __int64 v11; // r11
  unsigned int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 *v17; // rax
  unsigned int v18; // esi
  int v19; // eax
  int v20; // eax
  int v21; // eax
  __int64 *v22; // r10
  int v23; // edx
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-B8h]
  int v26; // [rsp+8h] [rbp-B8h]
  __int64 v27; // [rsp+18h] [rbp-A8h]
  __int64 v28; // [rsp+20h] [rbp-A0h]
  __int64 *v29; // [rsp+30h] [rbp-90h]
  __int64 v30; // [rsp+40h] [rbp-80h]
  __int64 *v31; // [rsp+50h] [rbp-70h]
  int v32[3]; // [rsp+5Ch] [rbp-64h] BYREF
  __int64 *v33; // [rsp+68h] [rbp-58h] BYREF
  __int64 v34; // [rsp+70h] [rbp-50h] BYREF
  __int64 *v35; // [rsp+78h] [rbp-48h]
  __int64 v36; // [rsp+80h] [rbp-40h]
  unsigned int v37; // [rsp+88h] [rbp-38h]

  v32[0] = a2;
  if ( a2 > 1 )
  {
    v2 = a1 + 136;
    if ( !(unsigned __int8)sub_1B977C0(a1 + 136, v32, &v34) )
    {
      v3 = sub_1B977C0(v2, v32, &v34);
      v28 = v34;
      if ( v3 )
      {
LABEL_5:
        v4 = *(_QWORD *)(a1 + 296);
        v29 = *(__int64 **)(v4 + 40);
        if ( *(__int64 **)(v4 + 32) == v29 )
          return;
        v31 = *(__int64 **)(v4 + 32);
        while ( 1 )
        {
          v30 = *v31;
          if ( (unsigned __int8)sub_1BF29F0(*(_QWORD *)(a1 + 320), *v31) )
          {
            v6 = *(_QWORD *)(v30 + 48);
            if ( v30 + 40 != v6 )
              break;
          }
LABEL_7:
          if ( v29 == ++v31 )
            return;
        }
        while ( 1 )
        {
          v7 = v6 - 24;
          if ( !v6 )
            v7 = 0;
          if ( (unsigned __int8)sub_1B91FD0(a1, v7) )
          {
            v34 = 0;
            v35 = 0;
            v36 = 0;
            v37 = 0;
            if ( !sub_1B92360((_DWORD *)a1, v7) && (int)sub_1BA7B00(a1, v7, (__int64)&v34, v32[0]) >= 0 )
            {
              v27 = v28 + 8;
              v8 = v35;
              if ( (_DWORD)v36 )
              {
                v9 = &v35[2 * v37];
                if ( v35 != v9 )
                {
                  while ( 1 )
                  {
                    v10 = v8;
                    if ( *v8 != -16 && *v8 != -8 )
                      break;
                    v8 += 2;
                    if ( v9 == v8 )
                      goto LABEL_16;
                  }
                  if ( v9 != v8 )
                  {
                    v11 = v6;
                    v12 = *(_DWORD *)(v28 + 32);
                    if ( !v12 )
                    {
LABEL_39:
                      ++*(_QWORD *)(v28 + 8);
LABEL_40:
                      v25 = v11;
                      v12 *= 2;
LABEL_41:
                      sub_14672C0(v27, v12);
                      sub_1463AD0(v27, v10, &v33);
                      v22 = v33;
                      v11 = v25;
                      v23 = *(_DWORD *)(v28 + 24) + 1;
                      goto LABEL_48;
                    }
                    while ( 1 )
                    {
                      v13 = *(_QWORD *)(v28 + 16);
                      v14 = (v12 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
                      v15 = (__int64 *)(v13 + 16LL * v14);
                      v16 = *v15;
                      if ( *v15 != *v10 )
                      {
                        v26 = 1;
                        v22 = 0;
                        while ( v16 != -8 )
                        {
                          if ( v16 != -16 || v22 )
                            v15 = v22;
                          v14 = (v12 - 1) & (v26 + v14);
                          v16 = *(_QWORD *)(v13 + 16LL * v14);
                          if ( *v10 == v16 )
                            goto LABEL_27;
                          ++v26;
                          v22 = v15;
                          v15 = (__int64 *)(v13 + 16LL * v14);
                        }
                        if ( !v22 )
                          v22 = v15;
                        v24 = *(_DWORD *)(v28 + 24);
                        ++*(_QWORD *)(v28 + 8);
                        v23 = v24 + 1;
                        if ( 4 * (v24 + 1) >= 3 * v12 )
                          goto LABEL_40;
                        if ( v12 - *(_DWORD *)(v28 + 28) - v23 <= v12 >> 3 )
                        {
                          v25 = v11;
                          goto LABEL_41;
                        }
LABEL_48:
                        *(_DWORD *)(v28 + 24) = v23;
                        if ( *v22 != -8 )
                          --*(_DWORD *)(v28 + 28);
                        *v22 = *v10;
                        *((_DWORD *)v22 + 2) = *((_DWORD *)v10 + 2);
                      }
LABEL_27:
                      v17 = v10 + 2;
                      if ( v9 == v10 + 2 )
                        break;
                      while ( 1 )
                      {
                        v10 = v17;
                        if ( *v17 != -8 && *v17 != -16 )
                          break;
                        v17 += 2;
                        if ( v9 == v17 )
                          goto LABEL_31;
                      }
                      if ( v9 == v17 )
                        break;
                      v12 = *(_DWORD *)(v28 + 32);
                      if ( !v12 )
                        goto LABEL_39;
                    }
LABEL_31:
                    v6 = v11;
                  }
                }
              }
            }
LABEL_16:
            sub_1412190(a1 + 64, v30);
            j___libc_free_0(v35);
          }
          v6 = *(_QWORD *)(v6 + 8);
          if ( v30 + 40 == v6 )
            goto LABEL_7;
        }
      }
      v18 = *(_DWORD *)(a1 + 160);
      v19 = *(_DWORD *)(a1 + 152);
      ++*(_QWORD *)(a1 + 136);
      v20 = v19 + 1;
      if ( 4 * v20 >= 3 * v18 )
      {
        v18 *= 2;
      }
      else if ( v18 - *(_DWORD *)(a1 + 156) - v20 > v18 >> 3 )
      {
LABEL_34:
        *(_DWORD *)(a1 + 152) = v20;
        if ( *(_DWORD *)v28 != -1 )
          --*(_DWORD *)(a1 + 156);
        v21 = v32[0];
        *(_QWORD *)(v28 + 8) = 0;
        *(_DWORD *)v28 = v21;
        *(_QWORD *)(v28 + 16) = 0;
        *(_QWORD *)(v28 + 24) = 0;
        *(_DWORD *)(v28 + 32) = 0;
        goto LABEL_5;
      }
      sub_1BA7500(v2, v18);
      sub_1B977C0(v2, v32, &v34);
      v28 = v34;
      v20 = *(_DWORD *)(a1 + 152) + 1;
      goto LABEL_34;
    }
  }
}
