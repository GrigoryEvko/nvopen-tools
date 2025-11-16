// Function: sub_FB6C00
// Address: 0xfb6c00
//
char __fastcall sub_FB6C00(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rcx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  int v7; // r14d
  unsigned int v8; // r15d
  char v9; // bl
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // r14
  int v17; // eax
  char v18; // r13
  unsigned int v19; // ebx
  __int64 *v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  char result; // al
  _QWORD *v29; // rdi
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // r13
  __int64 v37; // rdx
  __int64 *v38; // r13
  __int64 *v39; // rdx
  char v40; // [rsp+0h] [rbp-150h]
  __int64 *v41; // [rsp+0h] [rbp-150h]
  __int64 v42; // [rsp+8h] [rbp-148h]
  __int64 v43; // [rsp+10h] [rbp-140h]
  int v44; // [rsp+18h] [rbp-138h]
  char v45; // [rsp+18h] [rbp-138h]
  _QWORD *v46; // [rsp+28h] [rbp-128h] BYREF
  _QWORD *v47; // [rsp+30h] [rbp-120h]
  __int64 v48; // [rsp+38h] [rbp-118h]
  __int64 v49; // [rsp+40h] [rbp-110h]
  _QWORD v50[4]; // [rsp+50h] [rbp-100h] BYREF
  char v51; // [rsp+70h] [rbp-E0h]
  __int64 v52; // [rsp+80h] [rbp-D0h] BYREF
  __int64 *v53; // [rsp+88h] [rbp-C8h]
  __int64 v54; // [rsp+90h] [rbp-C0h]
  int v55; // [rsp+98h] [rbp-B8h]
  char v56; // [rsp+9Ch] [rbp-B4h]
  _BYTE v57[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)(a1 + 40);
  v43 = a2[5];
  v42 = v4;
  v5 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == v4 + 48 )
    goto LABEL_52;
  if ( !v5 )
    goto LABEL_50;
  v6 = v5 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
  {
LABEL_52:
    v52 = 0;
    v53 = (__int64 *)v57;
    v54 = 16;
    v55 = 0;
    v56 = 1;
  }
  else
  {
    v56 = 1;
    v52 = 0;
    v7 = sub_B46E30(v6);
    v53 = (__int64 *)v57;
    v54 = 16;
    v55 = 0;
    if ( v7 )
    {
      v8 = 0;
      v9 = 1;
      while ( 1 )
      {
        while ( 1 )
        {
          a2 = (_QWORD *)sub_B46EC0(v6, v8);
          if ( v9 )
            break;
LABEL_38:
          ++v8;
          sub_C8CC70((__int64)&v52, (__int64)a2, (__int64)v10, v11, v12, v13);
          v9 = v56;
          if ( v7 == v8 )
            goto LABEL_12;
        }
        v14 = v53;
        v10 = &v53[HIDWORD(v54)];
        if ( v53 == v10 )
        {
LABEL_40:
          if ( HIDWORD(v54) >= (unsigned int)v54 )
            goto LABEL_38;
          ++v8;
          ++HIDWORD(v54);
          *v10 = (__int64)a2;
          v9 = v56;
          ++v52;
          if ( v7 == v8 )
            break;
        }
        else
        {
          while ( a2 != (_QWORD *)*v14 )
          {
            if ( v10 == ++v14 )
              goto LABEL_40;
          }
          if ( v7 == ++v8 )
            break;
        }
      }
    }
  }
LABEL_12:
  v15 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v15 == v43 + 48 )
  {
LABEL_53:
    v18 = v56;
    result = 1;
    goto LABEL_29;
  }
  if ( !v15 )
LABEL_50:
    BUG();
  v16 = v15 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
    goto LABEL_53;
  v17 = sub_B46E30(v16);
  v18 = v56;
  v44 = v17;
  if ( v17 )
  {
    v40 = 0;
    v19 = 0;
    while ( 1 )
    {
      v46 = (_QWORD *)sub_B46EC0(v16, v19);
      a2 = v46;
      if ( v18 )
      {
        v20 = v53;
        v21 = &v53[HIDWORD(v54)];
        if ( v53 == v21 )
          goto LABEL_27;
        while ( 1 )
        {
          v22 = *v20;
          if ( v46 == (_QWORD *)*v20 )
            break;
          if ( v21 == ++v20 )
            goto LABEL_27;
        }
      }
      else
      {
        if ( !sub_C8CA60((__int64)&v52, (__int64)v46) )
          goto LABEL_35;
        v22 = (__int64)v46;
      }
      v47 = v50;
      v48 = 2;
      v50[0] = v42;
      v49 = 0;
      v50[1] = v43;
      v23 = sub_AA5930(v22);
      a2 = (_QWORD *)v24;
      result = sub_F915E0(v23, v24, v24, v25, v26, v27, v47, v48, v49);
      if ( result )
        goto LABEL_35;
      if ( !a3 )
      {
        v18 = v56;
        goto LABEL_29;
      }
      if ( *(_DWORD *)(a3 + 16) )
      {
        a2 = (_QWORD *)a3;
        sub_D6CB10((__int64)v50, a3, (__int64 *)&v46);
        v40 = v51;
        if ( v51 )
        {
          v34 = *(unsigned int *)(a3 + 40);
          v35 = (__int64)v46;
          if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
          {
            a2 = (_QWORD *)(a3 + 48);
            sub_C8D5F0(a3 + 32, (const void *)(a3 + 48), v34 + 1, 8u, v32, v33);
            v34 = *(unsigned int *)(a3 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8 * v34) = v35;
          ++*(_DWORD *)(a3 + 40);
LABEL_35:
          v18 = v56;
          goto LABEL_27;
        }
      }
      else
      {
        v29 = *(_QWORD **)(a3 + 32);
        a2 = &v29[*(unsigned int *)(a3 + 40)];
        if ( a2 == sub_F8ED40(v29, (__int64)a2, (__int64 *)&v46) )
        {
          v36 = (__int64)v46;
          if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
          {
            sub_C8D5F0(a3 + 32, (const void *)(a3 + 48), v30 + 1, 8u, v30, v31);
            a2 = (_QWORD *)(*(_QWORD *)(a3 + 32) + 8LL * *(unsigned int *)(a3 + 40));
          }
          *a2 = v36;
          v37 = (unsigned int)(*(_DWORD *)(a3 + 40) + 1);
          *(_DWORD *)(a3 + 40) = v37;
          if ( (unsigned int)v37 > 4 )
          {
            v38 = *(__int64 **)(a3 + 32);
            v41 = &v38[v37];
            do
            {
              v39 = v38;
              a2 = (_QWORD *)a3;
              ++v38;
              sub_D6CB10((__int64)v50, a3, v39);
            }
            while ( v41 != v38 );
          }
        }
      }
      v40 = 1;
      v18 = v56;
LABEL_27:
      if ( ++v19 == v44 )
      {
        result = v40 ^ 1;
        goto LABEL_29;
      }
    }
  }
  result = 1;
LABEL_29:
  if ( !v18 )
  {
    v45 = result;
    _libc_free(v53, a2);
    return v45;
  }
  return result;
}
