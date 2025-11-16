// Function: sub_2D701F0
// Address: 0x2d701f0
//
__int64 __fastcall sub_2D701F0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  unsigned int v3; // r14d
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v8; // r8
  __int64 v10; // r9
  char v11; // al
  __int64 v12; // r8
  int *v13; // r15
  int v14; // eax
  __int64 v15; // r15
  __int64 v16; // rax
  bool v17; // zf
  _DWORD *v18; // rbx
  __int64 v19; // r15
  unsigned int v20; // esi
  int v21; // eax
  int v22; // eax
  _QWORD *v23; // rdi
  __int64 v24; // rbx
  unsigned int v25; // esi
  int v26; // eax
  int v27; // eax
  _QWORD *v28; // rdi
  __int64 v29; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  int v33; // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+20h] [rbp-80h] BYREF
  __int64 v35; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v36[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v37; // [rsp+40h] [rbp-60h]
  _QWORD v38[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v39; // [rsp+60h] [rbp-40h]

  v4 = *a2;
  if ( *a2 == *a3 )
    return 0;
  v5 = a3[1];
  if ( a2[1] == v5 )
  {
    v8 = *a1;
    v37 = *a2;
    v36[0] = 0;
    v36[1] = 0;
    v10 = v8 + 728;
    if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
    {
      v29 = v8 + 728;
      v31 = v8;
      sub_BD73F0((__int64)v36);
      v10 = v29;
      v8 = v31;
    }
    v30 = v8;
    v32 = v10;
    v11 = sub_2D67BB0(v10, (__int64)v36, &v35);
    v12 = v30;
    if ( v11 )
    {
      v13 = (int *)(v35 + 24);
      goto LABEL_10;
    }
    v19 = v35;
    v38[0] = v35;
    v20 = *(_DWORD *)(v30 + 752);
    v21 = *(_DWORD *)(v30 + 744);
    ++*(_QWORD *)(v30 + 728);
    v22 = v21 + 1;
    if ( 4 * v22 >= 3 * v20 )
    {
      sub_2D6E640(v32, 2 * v20);
    }
    else
    {
      if ( v20 - *(_DWORD *)(v30 + 748) - v22 > v20 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(v12 + 744) = v22;
        if ( *(_QWORD *)(v19 + 16) != -4096 )
          --*(_DWORD *)(v12 + 748);
        v23 = (_QWORD *)v19;
        v13 = (int *)(v19 + 24);
        sub_2D57220(v23, v37);
        *v13 = 0;
LABEL_10:
        v14 = *v13;
        v15 = *a1;
        v38[0] = 0;
        v38[1] = 0;
        v33 = v14;
        v16 = *a3;
        v3 = v15 + 728;
        v17 = *a3 == -4096;
        v39 = v16;
        if ( v16 != 0 && !v17 && v16 != -8192 )
          sub_BD73F0((__int64)v38);
        if ( (unsigned __int8)sub_2D67BB0(v15 + 728, (__int64)v38, &v34) )
        {
          v18 = (_DWORD *)(v34 + 24);
LABEL_15:
          LOBYTE(v3) = v33 < *v18;
          sub_D68D70(v38);
          sub_D68D70(v36);
          return v3;
        }
        v24 = v34;
        v35 = v34;
        v25 = *(_DWORD *)(v15 + 752);
        v26 = *(_DWORD *)(v15 + 744);
        ++*(_QWORD *)(v15 + 728);
        v27 = v26 + 1;
        if ( 4 * v27 >= 3 * v25 )
        {
          v25 *= 2;
        }
        else if ( v25 - *(_DWORD *)(v15 + 748) - v27 > v25 >> 3 )
        {
LABEL_24:
          *(_DWORD *)(v15 + 744) = v27;
          if ( *(_QWORD *)(v24 + 16) != -4096 )
            --*(_DWORD *)(v15 + 748);
          v28 = (_QWORD *)v24;
          v18 = (_DWORD *)(v24 + 24);
          sub_2D57220(v28, v39);
          *v18 = 0;
          goto LABEL_15;
        }
        sub_2D6E640(v15 + 728, v25);
        sub_2D67BB0(v15 + 728, (__int64)v38, &v35);
        v24 = v35;
        v27 = *(_DWORD *)(v15 + 744) + 1;
        goto LABEL_24;
      }
      sub_2D6E640(v32, v20);
    }
    sub_2D67BB0(v32, (__int64)v36, v38);
    v12 = v30;
    v19 = v38[0];
    v22 = *(_DWORD *)(v30 + 744) + 1;
    goto LABEL_18;
  }
  LOBYTE(v3) = a2[1] < v5;
  return v3;
}
