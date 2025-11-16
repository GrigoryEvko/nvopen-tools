// Function: sub_13EE9C0
// Address: 0x13ee9c0
//
void __fastcall sub_13EE9C0(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // r13
  int v23; // edx
  int v24; // r9d
  __int64 i; // [rsp+10h] [rbp-A0h]
  __int64 v27; // [rsp+18h] [rbp-98h]
  int v28; // [rsp+20h] [rbp-90h] BYREF
  __int64 v29; // [rsp+28h] [rbp-88h]
  unsigned int v30; // [rsp+30h] [rbp-80h]
  __int64 v31; // [rsp+38h] [rbp-78h]
  unsigned int v32; // [rsp+40h] [rbp-70h]
  int v33; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+58h] [rbp-58h]
  unsigned int v35; // [rsp+60h] [rbp-50h]
  __int64 v36; // [rsp+68h] [rbp-48h]
  unsigned int v37; // [rsp+70h] [rbp-40h]

  v6 = a4;
  if ( !a4 )
  {
    v6 = a2;
    if ( *(_BYTE *)(a2 + 16) <= 0x17u )
      return;
  }
  v7 = *(_QWORD *)(a1 + 272);
  if ( !*(_BYTE *)(v7 + 184) )
  {
    sub_14CDF70(*(_QWORD *)(a1 + 272));
    v8 = *(unsigned int *)(v7 + 176);
    if ( !(_DWORD)v8 )
      goto LABEL_4;
LABEL_17:
    v18 = *(_QWORD *)(v7 + 160);
    v19 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v20 = v18 + 88LL * v19;
    v21 = *(_QWORD *)(v20 + 24);
    if ( a2 == v21 )
    {
LABEL_18:
      if ( v20 != v18 + 88 * v8 )
      {
        v22 = *(_QWORD *)(v20 + 40);
        for ( i = v22 + 32LL * *(unsigned int *)(v20 + 48); i != v22; v22 += 32 )
        {
          if ( *(_QWORD *)(v22 + 16) )
          {
            v27 = *(_QWORD *)(v22 + 16);
            if ( (unsigned __int8)sub_14AFF20(v27, v6, *(_QWORD *)(a1 + 288)) )
            {
              sub_13EE900(&v28, a2, *(_QWORD *)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF)), 1);
              sub_13EA210(&v33, (__int64)a3, (__int64)&v28);
              sub_13E8810(a3, (unsigned int *)&v33);
              if ( v33 == 3 )
              {
                if ( v37 > 0x40 && v36 )
                  j_j___libc_free_0_0(v36);
                if ( v35 > 0x40 && v34 )
                  j_j___libc_free_0_0(v34);
              }
              if ( v28 == 3 )
              {
                if ( v32 > 0x40 && v31 )
                  j_j___libc_free_0_0(v31);
                if ( v30 > 0x40 && v29 )
                  j_j___libc_free_0_0(v29);
              }
            }
          }
        }
      }
    }
    else
    {
      v23 = 1;
      while ( v21 != -8 )
      {
        v24 = v23 + 1;
        v19 = (v8 - 1) & (v23 + v19);
        v20 = v18 + 88LL * v19;
        v21 = *(_QWORD *)(v20 + 24);
        if ( a2 == v21 )
          goto LABEL_18;
        v23 = v24;
      }
    }
    goto LABEL_4;
  }
  v8 = *(unsigned int *)(v7 + 176);
  if ( (_DWORD)v8 )
    goto LABEL_17;
LABEL_4:
  v9 = sub_15F2050(v6);
  v10 = sub_15E0FD0(79);
  v12 = sub_16321A0(v9, v10, v11);
  if ( v12 )
  {
    if ( *(_QWORD *)(v12 + 8) )
    {
      v13 = *(_QWORD *)(v6 + 40);
      v14 = (_QWORD *)(v6 + 24);
      v15 = (_QWORD *)(v13 + 40);
      if ( v14 != (_QWORD *)(v13 + 40) )
      {
        while ( 1 )
        {
          if ( *((_BYTE *)v14 - 8) == 78 )
          {
            v16 = *(v14 - 6);
            if ( !*(_BYTE *)(v16 + 16) && *(_DWORD *)(v16 + 36) == 79 )
            {
              v17 = *(_QWORD *)(((unsigned __int64)(v14 - 3) & 0xFFFFFFFFFFFFFFF8LL)
                              - 24LL
                              * (*(_DWORD *)(((unsigned __int64)(v14 - 3) & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
              if ( v17 )
              {
                sub_13EE900(&v28, a2, v17, 1);
                sub_13EA210(&v33, (__int64)a3, (__int64)&v28);
                sub_13E8810(a3, (unsigned int *)&v33);
                if ( v33 == 3 )
                {
                  if ( v37 > 0x40 && v36 )
                    j_j___libc_free_0_0(v36);
                  if ( v35 > 0x40 && v34 )
                    j_j___libc_free_0_0(v34);
                }
                if ( v28 == 3 )
                {
                  if ( v32 > 0x40 && v31 )
                    j_j___libc_free_0_0(v31);
                  if ( v30 > 0x40 && v29 )
                    j_j___libc_free_0_0(v29);
                }
              }
            }
          }
          v14 = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v15 == v14 )
            break;
          if ( !v14 )
            BUG();
        }
      }
    }
  }
}
