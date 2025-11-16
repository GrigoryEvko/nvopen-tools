// Function: sub_1AD1560
// Address: 0x1ad1560
//
__int64 __fastcall sub_1AD1560(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 *v4; // r11
  __int64 *v5; // rdx
  __int64 *v6; // r15
  __int64 *v7; // rdx
  __int64 *v8; // r11
  __int64 v9; // rdx
  _QWORD *v10; // r15
  __int64 v11; // rcx
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 *v14; // r15
  __int64 *v15; // rcx
  __int64 *v16; // r15
  __int64 *v17; // rdx
  __int64 *v18; // r12
  __int64 *v19; // r13
  __int64 *v20; // r14
  __int64 *v21; // r15
  __int64 *v22; // rax
  __int64 *v23; // r14
  __int64 *v24; // r15
  __int64 *v25; // rbx
  __int64 *v26; // r12
  __int64 v27; // rsi
  __int64 *v28; // rbx
  __int64 v29; // rsi
  __int64 *v30; // [rsp+0h] [rbp-A0h]
  __int64 *v31; // [rsp+8h] [rbp-98h]
  __int64 *v32; // [rsp+10h] [rbp-90h]
  _QWORD *v33; // [rsp+18h] [rbp-88h]
  __int64 *v34; // [rsp+20h] [rbp-80h]
  __int64 *v35; // [rsp+28h] [rbp-78h]
  __int64 *v36; // [rsp+30h] [rbp-70h]
  __int64 *v37; // [rsp+38h] [rbp-68h]
  __int64 *v38; // [rsp+48h] [rbp-58h]
  __int64 *v39; // [rsp+50h] [rbp-50h]
  _QWORD *v40; // [rsp+58h] [rbp-48h]
  __int64 *v41; // [rsp+60h] [rbp-40h]
  __int64 *v42; // [rsp+68h] [rbp-38h]

  result = *(_QWORD *)a2;
  v3 = *(unsigned int *)(a2 + 8);
  *(_BYTE *)(a2 + 89) = 1;
  v36 = (__int64 *)(result + 8 * v3);
  if ( v36 != (__int64 *)result )
  {
    v4 = (__int64 *)result;
LABEL_3:
    result = *v4;
    ++*(_DWORD *)(result + 84);
    if ( *(_BYTE *)(result + 89) )
      goto LABEL_4;
    v5 = *(__int64 **)result;
    *(_BYTE *)(result + 89) = 1;
    result = (__int64)&v5[*(unsigned int *)(result + 8)];
    v38 = (__int64 *)result;
    if ( v5 == (__int64 *)result )
      goto LABEL_4;
    v30 = v4;
    v6 = v5;
    while ( 1 )
    {
      result = *v6;
      ++*(_DWORD *)(result + 84);
      if ( !*(_BYTE *)(result + 89) )
      {
        v7 = *(__int64 **)result;
        *(_BYTE *)(result + 89) = 1;
        result = (__int64)&v7[*(unsigned int *)(result + 8)];
        v39 = (__int64 *)result;
        if ( v7 != (__int64 *)result )
          break;
      }
LABEL_9:
      if ( v38 == ++v6 )
      {
        v4 = v30;
LABEL_4:
        if ( v36 == ++v4 )
          return result;
        goto LABEL_3;
      }
    }
    v31 = v6;
    v8 = v7;
    while ( 1 )
    {
      v9 = *v8;
      ++*(_DWORD *)(v9 + 84);
      if ( !*(_BYTE *)(v9 + 89) )
      {
        result = *(_QWORD *)v9;
        *(_BYTE *)(v9 + 89) = 1;
        v40 = (_QWORD *)(result + 8LL * *(unsigned int *)(v9 + 8));
        if ( (_QWORD *)result != v40 )
          break;
      }
LABEL_14:
      if ( v39 == ++v8 )
      {
        v6 = v31;
        goto LABEL_9;
      }
    }
    v32 = v8;
    v10 = (_QWORD *)result;
    while ( 1 )
    {
      v11 = *v10;
      ++*(_DWORD *)(v11 + 84);
      if ( !*(_BYTE *)(v11 + 89) )
      {
        v12 = *(__int64 **)v11;
        *(_BYTE *)(v11 + 89) = 1;
        v13 = *(unsigned int *)(v11 + 8);
        if ( v12 != &v12[v13] )
          break;
      }
LABEL_19:
      if ( v40 == ++v10 )
      {
        v8 = v32;
        goto LABEL_14;
      }
    }
    v41 = &v12[v13];
    v33 = v10;
    v14 = v12;
    while ( 1 )
    {
      result = *v14;
      ++*(_DWORD *)(result + 84);
      if ( !*(_BYTE *)(result + 89) )
      {
        v15 = *(__int64 **)result;
        *(_BYTE *)(result + 89) = 1;
        result = *(unsigned int *)(result + 8);
        if ( v15 != &v15[result] )
          break;
      }
LABEL_24:
      if ( v41 == ++v14 )
      {
        v10 = v33;
        goto LABEL_19;
      }
    }
    v42 = &v15[result];
    v34 = v14;
    v16 = v15;
    while ( 1 )
    {
      result = *v16;
      ++*(_DWORD *)(result + 84);
      if ( !*(_BYTE *)(result + 89) )
      {
        v17 = *(__int64 **)result;
        *(_BYTE *)(result + 89) = 1;
        result = *(unsigned int *)(result + 8);
        if ( v17 != &v17[result] )
          break;
      }
LABEL_29:
      if ( v42 == ++v16 )
      {
        v14 = v34;
        goto LABEL_24;
      }
    }
    v35 = v16;
    v18 = &v17[result];
    v19 = v17;
    while ( 1 )
    {
      result = *v19;
      ++*(_DWORD *)(result + 84);
      if ( !*(_BYTE *)(result + 89) )
      {
        v20 = *(__int64 **)result;
        *(_BYTE *)(result + 89) = 1;
        result = *(unsigned int *)(result + 8);
        v21 = &v20[result];
        if ( v20 != v21 )
          break;
      }
LABEL_34:
      if ( v18 == ++v19 )
      {
        v16 = v35;
        goto LABEL_29;
      }
    }
    v37 = v18;
    v22 = v20;
    v23 = v21;
    v24 = v22;
    while ( 1 )
    {
      result = *v24;
      ++*(_DWORD *)(result + 84);
      if ( !*(_BYTE *)(result + 89) )
      {
        v25 = *(__int64 **)result;
        *(_BYTE *)(result + 89) = 1;
        result = *(unsigned int *)(result + 8);
        if ( v25 != &v25[result] )
        {
          result = (__int64)&v25[result];
          v26 = v25;
          v27 = *v25;
          v28 = (__int64 *)result;
          ++*(_DWORD *)(v27 + 84);
          if ( !*(_BYTE *)(v27 + 89) )
            goto LABEL_45;
          while ( v28 != ++v26 )
          {
            v29 = *v26;
            ++*(_DWORD *)(v29 + 84);
            if ( !*(_BYTE *)(v29 + 89) )
LABEL_45:
              result = sub_1AD1560(a1);
          }
        }
      }
      if ( v23 == ++v24 )
      {
        v18 = v37;
        goto LABEL_34;
      }
    }
  }
  return result;
}
