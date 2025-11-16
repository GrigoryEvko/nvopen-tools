// Function: sub_2227C20
// Address: 0x2227c20
//
_QWORD *__fastcall sub_2227C20(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        int a5,
        int *a6,
        signed int a7,
        int a8,
        unsigned __int64 a9,
        __int64 a10,
        _DWORD *a11)
{
  unsigned int v13; // ebx
  __int64 v14; // rbp
  unsigned int v15; // eax
  __int64 v17; // r12
  unsigned int v18; // ebp
  unsigned __int64 v19; // r14
  int v20; // r13d
  bool v21; // r11
  char v22; // si
  char v23; // dl
  __int64 v24; // rsi
  unsigned __int8 v25; // al
  unsigned __int64 v26; // rax
  unsigned int *v27; // rax
  unsigned int v28; // eax
  int *v29; // rax
  int v30; // eax
  bool v31; // zf
  _QWORD *v32; // rax
  int v33; // eax
  _QWORD *v34; // r12
  unsigned __int64 v35; // r13
  int v36; // r14d
  int *v37; // rax
  int v38; // eax
  int v39; // eax
  char v43; // [rsp+1Ch] [rbp-3Ch]
  bool v44; // [rsp+1Dh] [rbp-3Bh]
  bool v45; // [rsp+1Fh] [rbp-39h]

  v13 = 10;
  v14 = sub_2243120(a10 + 208);
  if ( a9 != 2 )
  {
    v13 = 1000;
    if ( a9 != 4 )
      v13 = 1;
  }
  v15 = a3;
  v43 = a5 == -1;
  v17 = v14;
  v18 = v15;
  v19 = 0;
  v20 = 0;
  while ( 1 )
  {
    v21 = v18 == -1;
    v22 = v21 && a2 != 0;
    if ( v22 )
    {
      v37 = (int *)a2[2];
      if ( (unsigned __int64)v37 >= a2[3] )
      {
        v44 = v21 && a2 != 0;
        v38 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
        v21 = v18 == -1;
        v22 = v44;
      }
      else
      {
        v38 = *v37;
      }
      if ( v38 == -1 )
        a2 = 0;
      else
        v22 = 0;
    }
    else
    {
      v22 = v18 == -1;
    }
    v23 = v43 & (a4 != 0);
    if ( v23 )
      break;
    if ( v22 == v43 )
      goto LABEL_31;
LABEL_9:
    if ( v19 >= a9 )
      goto LABEL_31;
    if ( a2 && v21 )
    {
      v27 = (unsigned int *)a2[2];
      if ( (unsigned __int64)v27 >= a2[3] )
      {
        v28 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
        v24 = v28;
      }
      else
      {
        v24 = *v27;
        v28 = *v27;
      }
      if ( v28 == -1 )
        a2 = 0;
    }
    else
    {
      v24 = v18;
    }
    v25 = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v17 + 96LL))(v17, v24, 42) - 48;
    if ( v25 > 9u || (v20 = (char)v25 + 10 * v20, (int)(v20 * v13) > a8) || (int)(v13 + v20 * v13) <= a7 )
    {
      v39 = v20;
      v34 = a2;
      v35 = v19;
      v36 = v39;
      goto LABEL_40;
    }
    v26 = a2[2];
    v13 /= 0xAu;
    if ( v26 >= a2[3] )
      (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
    else
      a2[2] = v26 + 4;
    ++v19;
    v18 = -1;
  }
  v29 = (int *)a4[2];
  if ( (unsigned __int64)v29 >= a4[3] )
  {
    v45 = v21;
    v30 = (*(__int64 (**)(void))(*a4 + 72LL))();
    v21 = v45;
    v23 = v43 & (a4 != 0);
  }
  else
  {
    v30 = *v29;
  }
  v31 = v30 == -1;
  v32 = 0;
  if ( !v31 )
    v32 = a4;
  a4 = v32;
  if ( !v31 )
    v23 = 0;
  if ( v22 != v23 )
    goto LABEL_9;
LABEL_31:
  v33 = v20;
  v34 = a2;
  v35 = v19;
  v36 = v33;
  if ( v35 == a9 )
  {
    *a6 = v33;
    return v34;
  }
LABEL_40:
  if ( a9 == 4 && v35 == 2 )
    *a6 = v36 - 100;
  else
    *a11 |= 4u;
  return v34;
}
