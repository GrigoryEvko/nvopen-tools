// Function: sub_BA2FC0
// Address: 0xba2fc0
//
__int64 __fastcall sub_BA2FC0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rax
  int v4; // ebx
  __int64 v5; // r14
  int v6; // eax
  int v7; // r8d
  int v8; // r9d
  unsigned int i; // ebx
  __int64 *v10; // r15
  __int64 v11; // r12
  _BYTE *v12; // rax
  unsigned int v13; // ebx
  __int64 v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  unsigned int v22; // esi
  int v23; // eax
  _QWORD *v24; // rdx
  int v25; // eax
  _BYTE *v26; // [rsp+8h] [rbp-B8h]
  int v27; // [rsp+10h] [rbp-B0h]
  int v28; // [rsp+14h] [rbp-ACh]
  __int64 v29[2]; // [rsp+18h] [rbp-A8h] BYREF
  _QWORD *v30; // [rsp+28h] [rbp-98h] BYREF
  _QWORD *v31; // [rsp+30h] [rbp-90h] BYREF
  __int64 v32; // [rsp+38h] [rbp-88h] BYREF
  __int64 v33; // [rsp+40h] [rbp-80h] BYREF
  __int64 v34; // [rsp+48h] [rbp-78h] BYREF
  int v35; // [rsp+50h] [rbp-70h] BYREF
  __int64 v36; // [rsp+58h] [rbp-68h] BYREF
  __int8 v37; // [rsp+60h] [rbp-60h] BYREF
  __int8 v38[7]; // [rsp+61h] [rbp-5Fh] BYREF
  __int64 v39; // [rsp+68h] [rbp-58h] BYREF
  __int64 v40; // [rsp+70h] [rbp-50h]
  int v41; // [rsp+78h] [rbp-48h]
  __int64 v42[8]; // [rsp+80h] [rbp-40h] BYREF

  v29[0] = a1;
  v31 = *(_QWORD **)sub_A17150((_BYTE *)(a1 - 16));
  v32 = sub_AF5140(a1, 1u);
  v33 = sub_AF5140(a1, 5u);
  v34 = *((_QWORD *)sub_A17150((_BYTE *)(a1 - 16)) + 2);
  v35 = *(_DWORD *)(a1 + 16);
  v36 = *((_QWORD *)sub_A17150((_BYTE *)(a1 - 16)) + 3);
  v37 = *(_BYTE *)(a1 + 20);
  v38[0] = *(_BYTE *)(a1 + 21);
  v39 = *((_QWORD *)sub_A17150((_BYTE *)(a1 - 16)) + 6);
  v40 = *((_QWORD *)sub_A17150((_BYTE *)(a1 - 16)) + 7);
  v41 = *(_DWORD *)(a1 + 4);
  v3 = sub_A17150((_BYTE *)(a1 - 16));
  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 8);
  v42[0] = *((_QWORD *)v3 + 8);
  if ( v4 )
  {
    v6 = sub_AF8D50((__int64 *)&v31, &v32, &v33, &v34, &v35, &v36, &v37, v38, &v39, v42);
    v7 = v4 - 1;
    v8 = 1;
    for ( i = (v4 - 1) & v6; ; i = v7 & v13 )
    {
      v10 = (__int64 *)(v5 + 8LL * i);
      v11 = *v10;
      if ( *v10 == -4096 )
        break;
      if ( v11 != -8192 )
      {
        v27 = v8;
        v28 = v7;
        v26 = (_BYTE *)(v11 - 16);
        v12 = sub_A17150((_BYTE *)(v11 - 16));
        v7 = v28;
        v8 = v27;
        if ( v31 == *(_QWORD **)v12 )
        {
          v15 = sub_AF5140(v11, 1u);
          v7 = v28;
          v8 = v27;
          if ( v32 == v15 )
          {
            v16 = sub_AF5140(v11, 5u);
            v7 = v28;
            v8 = v27;
            if ( v33 == v16 )
            {
              v17 = sub_A17150(v26);
              v7 = v28;
              v8 = v27;
              if ( v34 == *((_QWORD *)v17 + 2) && v35 == *(_DWORD *)(v11 + 16) )
              {
                v18 = sub_A17150(v26);
                v7 = v28;
                v8 = v27;
                if ( v36 == *((_QWORD *)v18 + 3) && v37 == *(_BYTE *)(v11 + 20) && v38[0] == *(_BYTE *)(v11 + 21) )
                {
                  v19 = sub_A17150(v26);
                  v7 = v28;
                  v8 = v27;
                  if ( v39 == *((_QWORD *)v19 + 6) )
                  {
                    v20 = sub_A17150(v26);
                    v7 = v28;
                    v8 = v27;
                    if ( v40 == *((_QWORD *)v20 + 7) && v41 == *(_DWORD *)(v11 + 4) )
                    {
                      v21 = sub_A17150(v26);
                      v7 = v28;
                      v8 = v27;
                      if ( v42[0] == *((_QWORD *)v21 + 8) )
                      {
                        if ( v10 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
                          break;
                        return v11;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      v13 = v8 + i;
      ++v8;
    }
  }
  if ( !(unsigned __int8)sub_AFEC80(a2, v29, &v30) )
  {
    v22 = *(_DWORD *)(a2 + 24);
    v23 = *(_DWORD *)(a2 + 16);
    v24 = v30;
    ++*(_QWORD *)a2;
    v25 = v23 + 1;
    v31 = v24;
    if ( 4 * v25 >= 3 * v22 )
    {
      v22 *= 2;
    }
    else if ( v22 - *(_DWORD *)(a2 + 20) - v25 > v22 >> 3 )
    {
LABEL_24:
      *(_DWORD *)(a2 + 16) = v25;
      if ( *v24 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v24 = v29[0];
      return v29[0];
    }
    sub_B0B5C0(a2, v22);
    sub_AFEC80(a2, v29, &v31);
    v24 = v31;
    v25 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_24;
  }
  return v29[0];
}
