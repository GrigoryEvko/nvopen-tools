// Function: sub_226B280
// Address: 0x226b280
//
__int64 *__fastcall sub_226B280(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // r13
  void *v4; // rax
  __int64 v5; // rdi
  _BYTE *v6; // rax
  __int64 v8; // rax

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84050) )
  {
    v3 = *a2;
    *a2 = 0;
    v4 = sub_CB72A0();
    v5 = sub_CB6200((__int64)v4, *(unsigned __int8 **)(v3 + 8), *(_QWORD *)(v3 + 16));
    v6 = *(_BYTE **)(v5 + 32);
    if ( *(_BYTE **)(v5 + 24) == v6 )
    {
      sub_CB6200(v5, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v6 = 10;
      ++*(_QWORD *)(v5 + 32);
    }
    *a1 = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
    return a1;
  }
  else
  {
    v8 = *a2;
    *a2 = 0;
    *a1 = v8 | 1;
    return a1;
  }
}
