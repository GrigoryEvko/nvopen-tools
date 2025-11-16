// Function: sub_B10A70
// Address: 0xb10a70
//
__int64 __fastcall sub_B10A70(
        __int64 *a1,
        unsigned int a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        char a7)
{
  unsigned __int64 v7; // r10
  __int16 v8; // r15
  int v10; // r13d
  __int64 v12; // r9
  int v13; // ebx
  int v14; // eax
  int v15; // r11d
  int v16; // esi
  unsigned int i; // ecx
  __int64 v18; // rbx
  unsigned int v19; // ecx
  __int64 result; // rax
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // [rsp+0h] [rbp-A0h]
  __int64 v28; // [rsp+8h] [rbp-98h]
  unsigned __int64 v29; // [rsp+10h] [rbp-90h]
  unsigned int v30; // [rsp+24h] [rbp-7Ch]
  __int64 v31; // [rsp+28h] [rbp-78h]
  int v32; // [rsp+28h] [rbp-78h]
  __int64 v34; // [rsp+38h] [rbp-68h]
  __int64 v35; // [rsp+40h] [rbp-60h]
  __int64 v36; // [rsp+48h] [rbp-58h]
  unsigned __int64 v37; // [rsp+50h] [rbp-50h] BYREF
  __int64 v38; // [rsp+58h] [rbp-48h] BYREF
  __int64 v39[8]; // [rsp+60h] [rbp-40h] BYREF

  v7 = a4;
  v8 = a2;
  v10 = (int)a1;
  if ( a6 )
    goto LABEL_9;
  v12 = *a1;
  v37 = __PAIR64__(a3, a2);
  v38 = a4;
  v39[0] = a5;
  v13 = *(_DWORD *)(v12 + 1488);
  v34 = v12;
  v35 = *(_QWORD *)(v12 + 1472);
  if ( v13 )
  {
    v31 = a5;
    v14 = sub_AF7F90((int *)&v37, (int *)&v37 + 1, &v38, v39);
    v15 = v13 - 1;
    v7 = a4;
    a5 = v31;
    v16 = 1;
    for ( i = (v13 - 1) & v14; ; i = v15 & v19 )
    {
      v18 = *(_QWORD *)(v35 + 8LL * i);
      if ( v18 == -4096 )
        break;
      if ( v18 != -8192 && v37 == __PAIR64__(*(_DWORD *)(v18 + 4), *(unsigned __int16 *)(v18 + 2)) )
      {
        v27 = v35 + 8LL * i;
        v28 = a5;
        v29 = v7;
        v30 = i;
        v32 = v15;
        v25 = sub_A17150((_BYTE *)(v18 - 16));
        v15 = v32;
        i = v30;
        v7 = v29;
        a5 = v28;
        if ( v38 == *(_QWORD *)v25 )
        {
          v26 = sub_A17150((_BYTE *)(v18 - 16));
          v15 = v32;
          i = v30;
          v7 = v29;
          a5 = v28;
          if ( v39[0] == *((_QWORD *)v26 + 1) )
          {
            if ( v27 == *(_QWORD *)(v34 + 1472) + 8LL * *(unsigned int *)(v34 + 1488) )
              break;
            return v18;
          }
        }
      }
      v19 = v16 + i;
      ++v16;
    }
  }
  result = 0;
  if ( a7 )
  {
LABEL_9:
    v21 = *a1;
    v37 = v7;
    v38 = a5;
    v22 = v21 + 1464;
    v23 = sub_B97910(16, 2, a6);
    v24 = v23;
    if ( v23 )
    {
      v36 = v23;
      sub_B971C0(v23, v10, 32, a6, (unsigned int)&v37, 2, 0, 0);
      v24 = v36;
      *(_WORD *)(v36 + 2) = v8;
      *(_DWORD *)(v36 + 4) = a3;
    }
    return sub_B108B0(v24, a6, v22);
  }
  return result;
}
