// Function: sub_80C310
// Address: 0x80c310
//
__int64 __fastcall sub_80C310(__int64 *a1, __int64 *a2)
{
  int v2; // r13d
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned int v7; // edi
  _QWORD *v8; // rdi
  __int64 result; // rax
  _QWORD *v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  char v17; // [rsp+0h] [rbp-60h] BYREF
  char v18; // [rsp+1h] [rbp-5Fh]

  v2 = 0;
  v4 = *a1;
  if ( (*(_BYTE *)(v4 + 140) & 0xFB) == 8 )
    v2 = sub_8D4C10(v4, dword_4F077C4 != 2);
  v5 = qword_4F18BE0;
  v6 = *a2 + 2;
  if ( *((_DWORD *)a1 + 15) && !dword_4D0425C )
  {
    *a2 = v6;
    sub_8238B0(v5, "fL", 2);
    v11 = (unsigned int)(*((_DWORD *)a1 + 15) - 1);
    if ( (unsigned int)v11 > 9 )
    {
      v12 = (int)sub_622470(v11, &v17);
    }
    else
    {
      v18 = 0;
      v12 = 1;
      v17 = v11 + 48;
    }
    *a2 += v12;
    sub_8238B0(qword_4F18BE0, &v17, v12);
    v13 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    v14 = v13[2];
    if ( (unsigned __int64)(v14 + 1) > v13[1] )
    {
      sub_823810(v13);
      v13 = (_QWORD *)qword_4F18BE0;
      v14 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v13[4] + v14) = 112;
    ++v13[2];
  }
  else
  {
    *a2 = v6;
    sub_8238B0(v5, "fp", 2);
  }
  v7 = *((_DWORD *)a1 + 14);
  if ( v7 )
  {
    if ( v2 && !dword_4D0425C )
    {
      sub_80C190(v2, a2);
      v7 = *((_DWORD *)a1 + 14);
    }
    if ( v7 > 1 )
    {
      v15 = v7 - 2;
      if ( (unsigned int)v15 > 9 )
      {
        v16 = (int)sub_622470(v15, &v17);
      }
      else
      {
        v18 = 0;
        v16 = 1;
        v17 = v15 + 48;
      }
      *a2 += v16;
      sub_8238B0(qword_4F18BE0, &v17, v16);
    }
    v10 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    result = v10[2];
    if ( (unsigned __int64)(result + 1) > v10[1] )
    {
      sub_823810(v10);
      v10 = (_QWORD *)qword_4F18BE0;
      result = *(_QWORD *)(qword_4F18BE0 + 16);
      *(_BYTE *)(*(_QWORD *)(qword_4F18BE0 + 32) + result) = 95;
    }
    else
    {
      *(_BYTE *)(v10[4] + result) = 95;
    }
    ++v10[2];
  }
  else
  {
    v8 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    result = v8[2];
    if ( (unsigned __int64)(result + 1) > v8[1] )
    {
      sub_823810(v8);
      v8 = (_QWORD *)qword_4F18BE0;
      result = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v8[4] + result) = 84;
    ++v8[2];
  }
  return result;
}
