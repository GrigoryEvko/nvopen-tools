// Function: sub_2575FB0
// Address: 0x2575fb0
//
void __fastcall sub_2575FB0(_DWORD *a1, const void **a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v4; // r15
  __int64 v5; // rbx
  unsigned __int64 v6; // r14
  __int64 v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // ebx
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // rdi
  unsigned int *v15; // rdi
  unsigned __int64 v16; // r13
  __int64 v17; // [rsp+8h] [rbp-68h]
  _BYTE v18[96]; // [rsp+10h] [rbp-60h] BYREF

  v2 = (unsigned __int64)a2;
  if ( a1[4] )
  {
    sub_2575C40((__int64)v18, (__int64)a1, (__int64)a2);
    if ( v18[32] )
    {
      v10 = (unsigned int)a1[10];
      v11 = *((_QWORD *)a1 + 4);
      v12 = v10 + 1;
      v13 = a1[10];
      if ( v10 + 1 > (unsigned __int64)(unsigned int)a1[11] )
      {
        if ( v11 > v2 || v2 >= v11 + 16 * v10 )
        {
          sub_AE4800(a1 + 8, v12);
          v10 = (unsigned int)a1[10];
          v11 = *((_QWORD *)a1 + 4);
          v13 = a1[10];
        }
        else
        {
          v16 = v2 - v11;
          sub_AE4800(a1 + 8, v12);
          v11 = *((_QWORD *)a1 + 4);
          v10 = (unsigned int)a1[10];
          v2 = v11 + v16;
          v13 = a1[10];
        }
      }
      v14 = v11 + 16 * v10;
      if ( v14 )
      {
        sub_9865C0(v14, v2);
        v13 = a1[10];
      }
      a1[10] = v13 + 1;
    }
  }
  else
  {
    v4 = *((_QWORD *)a1 + 4);
    v17 = (unsigned int)a1[10];
    LODWORD(v5) = a1[10];
    v6 = v4 + 16 * v17;
    if ( v6 == sub_2546E70(v4, v6, a2) )
    {
      v7 = v17 + 1;
      if ( v17 + 1 > (unsigned __int64)(unsigned int)a1[11] )
      {
        v15 = a1 + 8;
        if ( v4 > v2 || v6 <= v2 )
        {
          sub_AE4800(v15, v7);
          v5 = (unsigned int)a1[10];
          v6 = *((_QWORD *)a1 + 4) + 16 * v5;
        }
        else
        {
          sub_AE4800(v15, v7);
          v2 = *((_QWORD *)a1 + 4) + v2 - v4;
          v5 = (unsigned int)a1[10];
          v6 = *((_QWORD *)a1 + 4) + 16 * v5;
        }
      }
      if ( v6 )
      {
        v8 = *(_DWORD *)(v2 + 8);
        *(_DWORD *)(v6 + 8) = v8;
        if ( v8 > 0x40 )
          sub_C43780(v6, (const void **)v2);
        else
          *(_QWORD *)v6 = *(_QWORD *)v2;
        LODWORD(v5) = a1[10];
      }
      v9 = v5 + 1;
      a1[10] = v9;
      if ( v9 > 8 )
        sub_2575D90((__int64)a1);
    }
  }
}
