// Function: sub_19D5B50
// Address: 0x19d5b50
//
__int64 __fastcall sub_19D5B50(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  unsigned int v7; // r15d
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r8
  unsigned __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  _QWORD *v17; // rcx
  __int64 v18; // rsi
  char v19; // dl
  unsigned __int8 v20; // al
  __int64 v22; // rdx
  char v23; // di
  unsigned int v24; // edx
  int v25; // edx
  int v26; // r9d
  __int64 v27; // rax
  unsigned __int64 v28; // rsi
  __int64 v29; // rax
  int v30; // eax
  int v31; // r14d
  unsigned int v32; // ebx
  __int64 v33; // rax
  unsigned __int64 v34; // r12
  _QWORD *v35; // rdi
  __int64 v36; // rax
  _QWORD *v37; // [rsp+0h] [rbp-90h]
  _QWORD *v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+28h] [rbp-68h]
  __int64 v41; // [rsp+30h] [rbp-60h]
  __int64 v42; // [rsp+38h] [rbp-58h]
  _QWORD *v43; // [rsp+48h] [rbp-48h] BYREF
  __int64 v44; // [rsp+50h] [rbp-40h] BYREF
  __int64 v45[7]; // [rsp+58h] [rbp-38h] BYREF

  if ( !*(_QWORD *)(a1 + 96) )
    sub_4263D6(a1, a2, a3);
  v7 = 0;
  v40 = (*(__int64 (__fastcall **)(__int64))(a1 + 104))(a1 + 80);
  v41 = a2 + 72;
  v42 = *(_QWORD *)(a2 + 80);
  if ( v42 != a2 + 72 )
  {
    while ( 1 )
    {
      v8 = 0;
      if ( v42 )
        v8 = v42 - 24;
      v9 = *(unsigned int *)(v40 + 48);
      if ( (_DWORD)v9 )
      {
        v10 = *(_QWORD *)(v40 + 32);
        v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v8 != *v12 )
        {
          v25 = 1;
          while ( v13 != -8 )
          {
            v26 = v25 + 1;
            v11 = (v9 - 1) & (v25 + v11);
            v12 = (__int64 *)(v10 + 16LL * v11);
            v13 = *v12;
            if ( v8 == *v12 )
              goto LABEL_7;
            v25 = v26;
          }
          goto LABEL_19;
        }
LABEL_7:
        if ( v12 != (__int64 *)(v10 + 16 * v9) )
        {
          if ( v12[1] )
          {
            v14 = *(_QWORD *)(v8 + 48);
            v15 = (_QWORD *)(v8 + 40);
            v16 = &v43;
            v43 = (_QWORD *)v14;
            if ( v8 + 40 != v14 )
              break;
          }
        }
      }
LABEL_19:
      v42 = *(_QWORD *)(v42 + 8);
      if ( v41 == v42 )
        return v7;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v17 = *(_QWORD **)(v14 + 8);
        v18 = v14 - 24;
        v43 = v17;
        v19 = *(_BYTE *)(v14 - 8);
        if ( v19 != 55 )
          break;
        v7 |= sub_19D5AF0(a1, v18, v16);
        v14 = (unsigned __int64)v43;
LABEL_12:
        if ( (_QWORD *)v14 == v15 )
          goto LABEL_19;
      }
      if ( v19 == 78 )
      {
        v22 = *(_QWORD *)(v14 - 48);
        if ( !*(_BYTE *)(v22 + 16) )
        {
          v23 = *(_BYTE *)(v22 + 33);
          if ( (v23 & 0x20) != 0 )
          {
            if ( *(_DWORD *)(v22 + 36) == 137 )
            {
              v24 = sub_19D1240((__int64 *)a1, v18, v16);
              goto LABEL_30;
            }
            if ( (v23 & 0x20) != 0 )
            {
              if ( *(_DWORD *)(v22 + 36) == 133 )
              {
                v24 = sub_19D3C80((__int64 *)a1, v18, a4, a5, a6);
                goto LABEL_30;
              }
              if ( (v23 & 0x20) != 0 && *(_DWORD *)(v22 + 36) == 135 )
              {
                v24 = sub_19D1AF0(a1, v18, v22);
LABEL_30:
                v14 = (unsigned __int64)v43;
                if ( (_BYTE)v24 )
                {
                  v7 = v24;
                  if ( v43 != *(_QWORD **)(v8 + 48) )
                  {
                    v14 = *v43 & 0xFFFFFFFFFFFFFFF8LL;
                    v43 = (_QWORD *)v14;
                  }
                }
                goto LABEL_12;
              }
            }
          }
        }
      }
      v44 = 0;
      v20 = *(_BYTE *)(v14 - 8);
      if ( v20 <= 0x17u )
        goto LABEL_18;
      if ( v20 == 78 )
      {
        v36 = v18;
        v28 = v18 & 0xFFFFFFFFFFFFFFF8LL;
        v29 = v36 | 4;
      }
      else
      {
        if ( v20 != 29 )
          goto LABEL_18;
        v27 = v18;
        v28 = v18 & 0xFFFFFFFFFFFFFFF8LL;
        v29 = v27 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v44 = v29;
      if ( v28 )
        break;
LABEL_18:
      v14 = (unsigned __int64)v17;
      if ( v17 == v15 )
        goto LABEL_19;
    }
    v30 = sub_165AFC0(&v44);
    if ( !v30 )
      goto LABEL_53;
    v39 = v8;
    v31 = v30;
    v38 = v15;
    v32 = 0;
    v37 = v16;
    while ( 1 )
    {
      v34 = v44 & 0xFFFFFFFFFFFFFFF8LL;
      v35 = (_QWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) + 56);
      if ( (v44 & 4) != 0 )
      {
        if ( (unsigned __int8)sub_1560290(v35, v32, 6) )
          goto LABEL_50;
        v33 = *(_QWORD *)(v34 - 24);
        if ( *(_BYTE *)(v33 + 16) )
          goto LABEL_45;
      }
      else
      {
        if ( (unsigned __int8)sub_1560290(v35, v32, 6) )
        {
LABEL_50:
          v7 |= sub_19D1CA0(a1, v44, v32);
          goto LABEL_45;
        }
        v33 = *(_QWORD *)(v34 - 72);
        if ( *(_BYTE *)(v33 + 16) )
          goto LABEL_45;
      }
      v45[0] = *(_QWORD *)(v33 + 112);
      if ( (unsigned __int8)sub_1560290(v45, v32, 6) )
        goto LABEL_50;
LABEL_45:
      if ( v31 == ++v32 )
      {
        v8 = v39;
        v15 = v38;
        v16 = v37;
LABEL_53:
        v17 = v43;
        goto LABEL_18;
      }
    }
  }
  return v7;
}
